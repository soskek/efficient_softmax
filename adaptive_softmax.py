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

from chainer import function_node
import chainer.functions
from chainer.utils import type_check

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check
from chainer import variable

"""
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
"""


def _broadcast_to(array, shape):
    if hasattr(numpy, "broadcast_to"):
        return numpy.broadcast_to(array, shape)
    dummy = numpy.empty(shape, array.dtype)
    return numpy.broadcast_arrays(array, dummy)[0]


def _check_class_weight_option(class_weight):
    if class_weight is not None:
        if class_weight.ndim != 1:
            raise ValueError('class_weight.ndim should be 1')
        if class_weight.dtype.kind != 'f':
            raise ValueError('The dtype of class_weight should be \'f\'')
        if isinstance(class_weight, variable.Variable):
            raise ValueError('class_weight should be a numpy.ndarray or '
                             'cupy.ndarray, not a chainer.Variable')


def _check_reduce_option(reduce):
    if reduce not in ('mean', 'no'):
        raise ValueError(
            "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)


def _check_input_values(x, t, ignore_label):
    # Extract the raw ndarray as Variable.__ge__ is not implemented.
    # We assume that t is already an ndarray.
    if isinstance(x, variable.Variable):
        x = x.data

    if not (((0 <= t) &
             (t < x.shape[1])) |
            (t == ignore_label)).all():
        msg = ('Each label `t` need to satisfy '
               '`0 <= t < x.shape[1] or t == %d`' % ignore_label)
        raise ValueError(msg)


class AdaptiveSoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    normalize = True

    def __init__(self, cutoff, normalize=True,
                 ignore_label=-1, reduce='mean'):
        self.cutoff = cutoff
        self.normalize = normalize
        self.ignore_label = ignore_label
        _check_reduce_option(reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() >= 4)
        x_type, t_type = in_types[:2]
        rest = len(in_types) - 2
        Ws_types = in_types[2: 2 + (rest - 1) // 2 + 1]
        Rs_types = in_types[2 + (rest - 1) // 2 + 1:]

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )
        for i in six.moves.range(len(Ws_types)):
            type_check.expect(
                x_type.dtype == Ws_types[i].dtype,
                x_type.shape[1] >= Ws_types[i].shape[1],
                Ws_types[i].ndim == 2,
            )
            print(i, len(Ws_types), len(Rs_types))
            if i != len(Ws_types) - 1:
                type_check.expect(
                    x_type.dtype == Rs_types[i].dtype,
                    x_type.shape[1] == Rs_types[i].shape[1],
                    x_type.shape[1] >= Rs_types[i].shape[0],
                    Rs_types[i].ndim == 2,
                )

    def linear(self, x, W):
        y = x.dot(W.T).astype(x.dtype, copy=False)
        return y

    def forward_cpu(self, inputs):
        x, t = inputs[:2]
        rest = len(inputs) - 2
        head_W, Ws = inputs[2], inputs[3:2 + (rest - 1) // 2 + 1]
        Rs = inputs[2 + (rest - 1) // 2 + 1:]
        n_tails = len(Rs)

        if chainer.is_debug():
            _check_input_values(x, t, self.ignore_label)

        self.retain_inputs(tuple(six.moves.range(len(inputs))))

        cluster_hots = []
        for i in six.moves.range(1, n_tails + 1):
            lower, upper = self.cutoff[i], self.cutoff[i + 1]
            in_cluster = numpy.logical_and(lower <= t, t < upper)
            cluster_hots.append(in_cluster)
        self.cluster_hots = cluster_hots

        head = self.linear(x, head_W)
        head = log_softmax._log_softmax(head)
        self.head = head
        tails = []
        for i, in_cluster in enumerate(cluster_hots, start=1):
            tail_idx = i - 1
            reduced_x = self.linear(x[in_cluster], Rs[tail_idx])
            out = self.linear(reduced_x, Ws[tail_idx])
            out = log_softmax._log_softmax(out)
            tails.append(out)
        self.tails = tails

        n_head_out = head_W.shape[0] - n_tails
        n_out = n_head_out + sum(W.shape[0] for W in Ws)
        shape = (x.shape[0], n_out)
        log_y = numpy.full(shape, numpy.nan, dtype=x.dtype)
        # for error check, nan is filled.

        log_y[:, :n_head_out] = head[:, :n_head_out]
        # it is possible ``log_[in_head, :n_headout]``, maybe faster?
        for i, (in_cluster, tail) in enumerate(
                zip(cluster_hots, tails), start=1):
            lower, upper = self.cutoff[i], self.cutoff[i + 1]

            tail_main = head[in_cluster, n_head_out + i - 1:n_head_out + i]
            tail_main = numpy.broadcast_to(tail_main, tail.shape)
            log_y[in_cluster, lower:upper] = tail_main + tail
            # not_in_cluster = numpy.logical_not(in_cluster)
            # log_y[not_in_cluster, lower] = tail_main[not_in_cluster]
            # These are not required,
            # because we need just picked values by t (after log softmax).

        self.y = numpy.exp(log_y)
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)
        log_p = log_yd[numpy.maximum(t.ravel(), 0), numpy.arange(t.size)]

        log_p *= (t.ravel() != self.ignore_label)
        if self.reduce == 'mean':
            # deal with the case where the SoftmaxCrossEntropy is
            # unpickled from the old version
            if self.normalize:
                count = (t != self.ignore_label).sum()
            else:
                count = len(x)
            self._coeff = 1.0 / max(count, 1)

            y = log_p.sum(keepdims=True) * (-self._coeff)
            return y.reshape(()),
        else:
            return -log_p.reshape(t.shape),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        if chainer.is_debug():
            _check_input_values(x, t, self.ignore_label)

        log_y = log_softmax._log_softmax(x)
        self.y = cupy.exp(log_y)

        if self.normalize:
            coeff = cupy.maximum(1, (t != self.ignore_label).sum())
        else:
            coeff = max(1, len(t))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        if self.reduce == 'mean':
            ret = cuda.reduce(
                'S t, raw T log_y, int32 n_channel, raw T coeff, '
                'S ignore_label',
                'T out',
                't == ignore_label ? T(0) : log_y[_j * n_channel + t]',
                'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1],
              self._coeff, self.ignore_label)
        else:
            ret = cuda.elementwise(
                'S t, raw T log_y, int32 n_channel, T ignore', 'T out',
                '''
                if (t == ignore) {
                  out = 0;
                } else {
                  out = -log_y[i * n_channel + t];
                }
                ''',
                'softmax_crossent_no_reduce_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1], self.ignore_label)
            ret = ret.reshape(t.shape)
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs[:2]
        rest = len(inputs) - 2
        head_W, Ws = inputs[2], inputs[3:2 + (rest - 1) // 2 + 1]
        Rs = inputs[2 + (rest - 1) // 2 + 1:]
        n_tails = len(Rs)

        gloss = grad_outputs[0]
        y = self.y.copy()

        gx = y
        gx[numpy.arange(len(t)), numpy.maximum(t, 0)] -= 1

        gx *= (t != self.ignore_label).reshape((len(t), 1))

        if self.reduce == 'mean':
            gx *= gloss * self._coeff
        else:
            gx *= gloss[:, None]

        # add processing

        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        if hasattr(self, 'y'):
            y = self.y
        else:
            y = log_softmax._log_softmax(x)
            cupy.exp(y, out=y)
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        if self.reduce == 'mean':
            coeff = gloss * self._coeff
        else:
            coeff = gloss[:, None, ...]

        gx = cuda.elementwise(
            'T y, S t, T coeff, S n_channel, S n_unit, S ignore_label',
            'T gx',
            '''
              const int c = (i / n_unit % n_channel);
              gx = t == ignore_label ? 0 : coeff * (y - (c == t));
            ''',
            'softmax_crossent_bwd')(
                y, cupy.expand_dims(t, 1), coeff, x.shape[1],
                n_unit, self.ignore_label)

        return gx, None


def adaptive_softmax_cross_entropy(
        x, t, Ws, Rs, cutoff, normalize=True,
        ignore_label=-1, reduce='mean', enable_double_backprop=False):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a multidimensional array whose element indicates
            hidden states: the first axis of the variable
            represents the number of samples, and the second axis represents
            the number of hidden units.
        Ws (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variables of weight matrices for word outputs.
            The first matrix is for the head.
            The rest matrices are for the tails in order.
        Rs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variables of weight matrices for reducing hidden units.
            The matrices are for the tails in order.
            The number of matrices must be ``len(Ws) - 1``.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding an :class:`numpy.int32` vector of ground truth
            labels. If ``t[i] == ignore_label``, corresponding ``x[i]`` is
            ignored.
        cutoff (list of int):
            Cutoff indices of clusters. e.g. [0, 2000, 10000, n_vocab]
        normalize (bool): If ``True``, this function normalizes the cross
            entropy loss across all instances. If ``False``, it only
            normalizes along a batch size.
        ignore_label (int): Label value you want to ignore. Its default value
            is ``-1``. See description of the argument `t`.
        reduce (str): A string that determines whether to reduce the loss
            values. If it is ``'mean'``, it computes the sum of the individual
            cross entropy and normalize it according to ``normalize`` option.
            If it is ``'no'``, this function computes cross entropy for each
            instance and does not normalize it (``normalize`` option is
            ignored). In this case, the loss value of the ignored instance,
            which has ``ignore_label`` as its target value, is set to ``0``.
        enable_double_backprop (bool): If ``True``, this function uses
            implementation that supports higher order differentiation.
            If ``False``, it uses single-backprop implementation.
            This function use the single-backprop version because we expect
            it is faster. So, if you need second or higher derivatives,
            you need to turn it on explicitly.

    Returns:
        ~chainer.Variable: A variable holding a scalar array of the cross
        entropy loss.  If ``reduce`` is ``'mean'``, it is a scalar array.
        If ``reduce`` is ``'no'``, the shape is same as that of ``x``.

    .. note::

       This function is differentiable only by ``x``.

    .. admonition:: Example

        >>> x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]]).astype('f')
        >>> x
        array([[-1.,  0.,  1.,  2.],
               [ 2.,  0.,  1., -1.]], dtype=float32)
        >>> t = np.array([3, 0]).astype('i')
        >>> t
        array([3, 0], dtype=int32)
        >>> y = F.adaptive_softmax_cross_entropy(x, t)
        >>> y
        variable(0.4401897192001343)
        >>> log_softmax = -F.log_softmax(x)
        >>> expected_loss = np.mean([log_softmax[row, column].data \
for row, column in enumerate(t)])
        >>> y.array == expected_loss
        True

    """

    if enable_double_backprop:
        raise NotImplementedError()
    else:
        return AdaptiveSoftmaxCrossEntropy(
            cutoff, normalize, ignore_label, reduce)(
                x, t, *Ws, *Rs)


class AdaptiveSoftmaxOutputLayer(chainer.Chain):
    def __init__(self, n_units, n_vocab,
                 cutoff=[2000, 10000], reduce_k=4):
        super(AdaptiveSoftmaxOutputLayer, self).__init__()
        assert(all(c < n_vocab - 1 for c in cutoff))
        self.n_clusters = len(cutoff) + 1
        self.n_tails = self.n_clusters - 1
        cutoff.append(n_vocab)
        with self.init_scope():
            self.head = variable.Parameter(None)
            self.head.initialize((cutoff[0] + self.n_tails, n_units))
            # self.head = L.Linear(n_units, cutoff[0] + self.n_tails)
            tail_units = n_units
            for i in range(1, self.n_tails + 1):
                tail_units = tail_units // reduce_k
                n_comp_words = cutoff[i] - cutoff[i - 1]
                assert(tail_units > 0)
                assert(n_comp_words > 0)
                # TODO: reduce at once: d -> [d//4 + d//16 + ...] and split
                # as far as we do not use batch reduction (B -> pB) for tails
                """
                self.add_link('reduce{}'.format(i),
                              L.Linear(n_units, tail_units))
                self.add_link('tail{}'.format(i),
                              L.Linear(tail_units, n_comp_words))
                """

                self.add_param('reduce{}'.format(i))
                getattr(self, 'reduce{}'.format(i)).initialize(
                    (tail_units, n_units))
                self.add_param('tail{}'.format(i))
                getattr(self, 'tail{}'.format(i)).initialize(
                    (n_comp_words, tail_units))

            cutoff = self.xp.array([0] + cutoff, dtype=np.int32)
            assert(len(cutoff) == self.n_clusters + 1)
            self.add_param('cutoff', cutoff.shape, dtype='i')
            self.cutoff.data[:] = cutoff
        print('init adaptive softmax')

    def output_and_loss(self, h, t):
        Ws = [self.head] + [getattr(self, 'tail{}'.format(i))
                            for i in range(1, self.n_tails + 1)]
        Rs = [getattr(self, 'reduce{}'.format(i))
              for i in range(1, self.n_tails + 1)]
        cutoff = self.cutoff.data
        return adaptive_softmax_cross_entropy(
            h, t, Ws, Rs, cutoff, normalize=False, reduce='mean')
