import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn

import argparse
import time
import os

from utils import TextLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                   help='data directory containing input.txt')
parser.add_argument('--train_dir', type=str, default='models',
                   help='model directory to store checkpointed models')
parser.add_argument('--rnn_size', type=int, default=128,
                   help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,
                   help='number of layers in the RNN')
parser.add_argument('--model', type=str, default='rnn',
                   help='rnn, gru, or lstm')
parser.add_argument('--batch_size', type=int, default=20,
                   help='minibatch size')
parser.add_argument('--seq_length', type=int, default=20,
                   help='RNN sequence length')
parser.add_argument('--num_epochs', type=int, default=20,
                   help='number of epochs')
parser.add_argument('--save_every', type=int, default=1000,
                   help='save frequency')
parser.add_argument('--grad_clip', type=float, default=5.,
                   help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.01,
                   help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.95,
                   help='decay rate for rmsprop')


args = parser.parse_args()

loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)

if args.model == 'rnn':
    cell_fn = rnn_cell.BasicRNNCell
elif args.model == 'gru':
    cell_fn = rnn_cell.GRUCell
elif args.model == 'lstm':
    cell_fn = rnn_cell.BasicLSTMCell
else:
    raise Exception("model type not supported: {}".format(args.model))

cell = cell_fn(args.rnn_size)

cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
initial_state = cell.zero_state(args.batch_size, tf.float32)

with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [loader.vocab_size, args.rnn_size])
      inputs = tf.split(
          1, args.seq_length, tf.nn.embedding_lookup(embedding, input_data))
      inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

outputs, states = seq2seq.rnn_decoder(inputs, initial_state, cell)
output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
logits = tf.nn.xw_plus_b(output,
        tf.get_variable("softmax_w", [args.rnn_size, loader.vocab_size]),
        tf.get_variable("softmax_b", [loader.vocab_size]))
loss = seq2seq.sequence_loss_by_example([logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([args.batch_size * args.seq_length])],
        loader.vocab_size)
cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
final_state = states[-1]
lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
        args.grad_clip)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    for e in xrange(args.num_epochs):
        sess.run(tf.assign(lr, args.learning_rate))
        loader.reset_batch_pointer()
        state = initial_state.eval()
        for b in xrange(loader.num_batches):
            start = time.time()
            x, y = loader.next_batch()
            feed = {input_data: x, targets: y, initial_state: state}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
            end = time.time()
            print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(e * loader.num_batches + b,
                        args.num_epochs * loader.num_batches,
                        e, train_loss, end - start)
            if (e * loader.num_batches + b) % args.save_every == 0:
                checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = e * loader.num_batches + b)
                print "model saved to {}".format(checkpoint_path)


