# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`.
Every 1000 iterations, sampled output from the RNN will be printed.

# Roadmap
- Model serialization functionality, i.e. save and load models from disk
- Add explanatory comments
- Expose more command-line arguments
- Compare accuracy and performance with char-rnn
