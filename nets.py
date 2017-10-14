#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import black_out


class GradientMultiplier(chainer.function.Function):

    """Gradient Multiplier."""

    def __init__(self, coefficient):
        self.coefficient = coefficient[None]

    def forward_cpu(self, x):
        return x[0],

    def forward_gpu(self, x):
        return x[0],

    def backward_cpu(self, x, gy):
        return chainer.utils.force_array(gy[0] * self.coefficient),

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T c, T gy', 'T gx',
            'gx = gy * c',
            'gradmul_bwd')(self.coefficient, gy[0])
        return gx,


def gradient_multiplier(x, coefficient=1.):
    """Gradient Multiplier function.
    .. math:: f(x)=x. f'(x)=coefficient.
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.
    """
    return GradientMultiplier(coefficient)(x)


def embed_seq_batch(embed, seq_batch, dropout=0.):
    batchsize = len(seq_batch)
    e_seq_batch = F.split_axis(
        F.dropout(embed(F.concat(seq_batch, axis=0)), ratio=dropout),
        batchsize, axis=0)
    # [(len, ), ] x batchsize
    return e_seq_batch


class AdaptiveSoftmaxOutputLayer(chainer.Chain):
    def __init__(self, n_units, n_vocab,
                 cutoff=[2000, 10000], reduce_k=4):
        super(AdaptiveSoftmaxOutputLayer, self).__init__()
        assert(all(c < n_vocab - 1 for c in cutoff))
        self.n_clusters = len(cutoff) + 1
        self.n_tails = self.n_clusters - 1
        cutoff.append(n_vocab - 1)
        with self.init_scope():
            self.head = L.Linear(n_units, cutoff[0] + self.n_tails)
            tail_units = n_units
            for i in range(1, self.n_tails + 1):
                tail_units = tail_units // reduce_k
                n_comp_words = cutoff[i] - cutoff[i - 1]
                assert(tail_units > 0)
                assert(n_comp_words > 0)
                # TODO: reduce at once: d -> [d//4 + d//16 + ...] and split
                # as far as we do not use batch reduction (B -> pB) for tails
                self.add_link('reduce{}'.format(i),
                              L.Linear(n_units, tail_units))
                self.add_link('tail{}'.format(i),
                              L.Linear(tail_units, n_comp_words))

            cutoff = self.xp.array([0] + cutoff, dtype=np.int32)
            assert(len(cutoff) == self.n_clusters + 1)
            self.add_param('cutoff', cutoff.shape, dtype='f')
            self.cutoff.data[:] = cutoff
        print('init adaptive softmax')

    def output_and_loss(self, h, t):
        xp = self.xp
        """
        cluster2hots = []
        t_data = t.data if isinstance(t, chainer.Variable) else t
        for i in range(self.n_clusters):
            in_cluster = xp.logical_and(self.cutoff[i] <= t_data,
                                        t_data < self.cutoff[i + 1])
            cluster2hots.append(in_cluster)
        """
        head_logit = self.head(h)
        # tail_logits = [head_logit[:, i - self.n_tails: i - self.n_tails + 1]
        #                for i in range(self.n_tail)]
        cluster2logit = [head_logit[:, :- self.n_tails]]
        for i in range(1, self.n_tails + 1):
            reduced_h = getattr(self, 'reduce{}'.format(i))(h)
            comp_logit = getattr(self, 'tail{}'.format(i))(reduced_h)
            cluster2logit.append(comp_logit)

        logit = F.concat(cluster2logit, axis=1)
        return F.softmax_cross_entropy(
            logit, t, normalize=False, reduce='mean')


class BlackOutOutputLayer(black_out.BlackOut):
    def output_and_loss(self, h, t):
        if chainer.config.train:
            return super(BlackOutOutputLayer, self).__call__(h, t)
        else:
            logit = self(h)
            return F.softmax_cross_entropy(logit, t, normalize=False, reduce='mean')

    def __call__(self, h):
        return F.linear(h, self.W)


class NormalOutputLayer(L.Linear):
    def __init__(self, n_units, n_vocab, dead_or_alive=False):
        super(NormalOutputLayer, self).__init__(n_units, n_vocab)
        self.dead_or_alive = dead_or_alive

    def output_and_loss(self, h, t):
        logit = self(h)
        # if hasattr(self, 'word_weight'):
        #    logit = gradient_multiplier(logit, getattr(self, 'word_weight'))
        if self.dead_or_alive and hasattr(self, 'word_weight'):
            valid = self.xp.broadcast_to(
                (self.word_weight > 0.)[None], logit.shape)
            logit = F.where(
                valid, logit,
                self.xp.full(logit.shape, -1024., dtype=np.float32))
        return F.softmax_cross_entropy(
            logit, t, normalize=False,
            class_weight=getattr(self, 'word_weight', None), reduce='mean')


class SharedOutputLayer(chainer.Chain):
    def __init__(self, W, bias=True, scale=True, dead_or_alive=False):
        super(SharedOutputLayer, self).__init__()
        self.W = W
        self.dead_or_alive = dead_or_alive
        with self.init_scope():
            if bias:
                self.add_param('b', (W.shape[0], ), dtype='f')
                self.b.data[:] = 0.
            else:
                self.b = None
            if scale:
                self.add_param('scale', (1, ), dtype='f')
                self.scale.data[:] = 1.
            else:
                self.scale = None

    def output_and_loss(self, h, t):
        logit = self(h)
        # if hasattr(self, 'word_weight'):
        #    logit = gradient_multiplier(logit, getattr(self, 'word_weight'))
        if self.dead_or_alive and hasattr(self, 'word_weight'):
            valid = self.xp.broadcast_to(
                (self.word_weight > 0.)[None], logit.shape)
            logit = F.where(
                valid, logit,
                self.xp.full(logit.shape, -1024., dtype=np.float32))
        return F.softmax_cross_entropy(
            logit, t, normalize=False,
            class_weight=getattr(self, 'word_weight', None), reduce='mean')

    def __call__(self, x):
        out = F.linear(x, self.W, self.b)
        if self.scale is not None:
            out *= F.broadcast_to(self.scale[None], out.shape)
        return out

# Definition of a recurrent net for language modeling


class RNNForLM(chainer.Chain):
    # TODO: nstep LSTM
    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5,
                 share_embedding=False, blackout_counts=None,
                 dead_or_alive=False, adaptive_softmax=False):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.rnn = L.NStepLSTM(n_layers, n_units, n_units, dropout)
            assert(not (share_embedding and blackout_counts is not None))
            if share_embedding:
                self.output = SharedOutputLayer(self.embed.W,
                                                dead_or_alive=dead_or_alive)
            elif blackout_counts is not None:
                sample_size = max(500, (n_vocab // 200))
                self.output = BlackOutOutputLayer(
                    n_units, blackout_counts, sample_size)
                print('Blackout sample size is {}'.format(sample_size))
            elif adaptive_softmax:
                self.output = AdaptiveSoftmaxOutputLayer(
                    n_units, n_vocab,
                    cutoff=[2000, 10000], reduce_k=4)
            else:
                self.output = NormalOutputLayer(n_units, n_vocab,
                                                dead_or_alive=dead_or_alive)
        self.dropout = dropout
        self.n_units = n_units
        self.n_layers = n_layers

        for name, param in self.namedparams():
            if param.ndim != 1:
                # This initialization is applied only for weight matrices
                param.data[...] = np.random.uniform(-0.1,
                                                    0.1, param.data.shape)

        self.loss = 0.
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, x):
        raise NotImplementedError()

    def call_rnn(self, e_seq_batch):
        batchsize = len(e_seq_batch)
        if self.h is None:
            self.h = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        if self.c is None:
            self.c = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        self.h, self.c, y_seq_batch = self.rnn(self.h, self.c, e_seq_batch)
        return y_seq_batch

    def encode_seq_batch(self, x_seq_batch):
        e_seq_batch = embed_seq_batch(
            self.embed, x_seq_batch, dropout=self.dropout)
        y_seq_batch = self.call_rnn(e_seq_batch)
        return y_seq_batch

    def forward_seq_batch(self, x_seq_batch, t_seq_batch, normalize=None):
        y_seq_batch = self.encode_seq_batch(x_seq_batch)
        loss = self.output_and_loss_from_seq_batch(
            y_seq_batch, t_seq_batch, normalize)
        return loss

    def output_and_loss_from_seq_batch(self, y_seq_batch, t_seq_batch, normalize=None):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        t = F.concat(t_seq_batch, axis=0)
        loss = self.output.output_and_loss(y, t)
        if normalize is not None:
            loss *= 1. * t.shape[0] / normalize
        else:
            loss *= t.shape[0]
        return loss

    def output_from_seq_batch(self, y_seq_batch):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        return self.output(y)

    def pop_loss(self):
        loss = self.loss
        self.loss = 0.
        return loss
