import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='models',
                       help='model directory to store checkpointed models')
    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.train_dir, 'config.pkl')) as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.train_dir, 'vocab.pkl')) as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print model.sample(sess, chars, vocab)

if __name__ == '__main__':
    main()
