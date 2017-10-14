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
