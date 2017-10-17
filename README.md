# Efficient Softmax Approximation

Implementations of Blackout and Adaptive Softmax for efficiently calculating word distribution for language modeling of very large vocabularies.

LSTM language models are derived from [rnnlm_chainer](https://github.com/soskek/rnnlm_chainer).

Available output layers are as follows

- Linear + softmax with cross entropy loss. A usual output layer.
- `--share-embedding`: A variant using the word embedding matrix shared with the input layer for the output layer.
- `--adaptive-softmax`: [Adaptive softmax](http://proceedings.mlr.press/v70/grave17a/grave17a.pdf)
- `--blackout`: [BlackOut](https://arxiv.org/pdf/1511.06909.pdf) (BlackOut is not faster on GPU.)

### Adaptive Softmax

- Efficient softmax approximation for GPUs
- Edouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou, ICML 2017
- [paper](http://proceedings.mlr.press/v70/grave17a/grave17a.pdf)
- [authors' Lua code](https://github.com/facebookresearch/adaptive-softmax)

### BlackOut

- BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies
- Shihao Ji, S. V. N. Vishwanathan, Nadathur Satish, Michael J. Anderson, Pradeep Dubey, ICLR 2016
- [paper](https://arxiv.org/pdf/1511.06909.pdf)
- [authors' C++ code](https://github.com/IntelLabs/rnnlm)

# How to Run

```
python -u train.py -g 0
```

## Datasets

- PennTreeBank
- Wikitext-2
- Wikitext-103

For wikitext, run `prepare_wikitext.sh` for downloading the datasets.
