from chainer.functions.array import broadcast
from chainer.functions.array import concat
from chainer.functions.array import expand_dims
from chainer.functions.array import reshape
from chainer.functions.connection import embed_id
from chainer.functions.math import average
from chainer.functions.math import exponential
from chainer.functions.math import logsumexp
from chainer.functions.math import matmul
from chainer.functions.math import sum as _sum

from chainer import cuda
from chainer import functions as F
import numpy as np

import numpy

import chainer
from chainer import cuda
from chainer import link
from chainer.utils import walker_alias
from chainer import variable

def black_out(x, t, W, log_q, samples, reduce='mean'):
    """BlackOut loss function.

    BlackOut loss function is defined as

    .. math::

      -\\log(p(t)) - \\sum_{s \\in S} \\log(1 - p(s)),

    where :math:`t` is the correct label, :math:`S` is a set of negative
    examples and :math:`p(\cdot)` is likelihood of a given label.
    And, :math:`p` is defined as

    .. math::

       p(y) = \\frac{\\exp(W_y^\\top x)}{
       \\sum_{s \\in samples} \\exp(W_s^\\top x)}.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the
    no loss values. If it is ``'mean'``, this function takes
    a mean of loss values.

    Args:
        x (~chainer.Variable): Batch of input vectors.
            Its shape should be :math:`(N, D)`.
        t (~chainer.Variable): Vector of ground truth labels.
            Its shape should be :math:`(N,)`. Each elements :math:`v`
            should satisfy :math:`0 \geq v \geq V` or :math:`-1`
            where :math:`V` is the number of label types.
        W (~chainer.Variable): Weight matrix.
            Its shape should be :math:`(V, D)`
        samples (~chainer.Variable): Negative samples.
            Its shape should be :math:`(N, S)` where :math:`S` is
            the number of negative samples.
        reduce (str): Reduction option. Its value must be either
            ``'no'`` or ``'mean'``. Otherwise,
            :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable object holding loss value(s).
            If ``reduce`` is ``'no'``, the output variable holds an
            array whose shape is :math:`(N,)` .
            If it is ``'mean'``, it holds a scalar.

    See: `BlackOut: Speeding up Recurrent Neural Network Language Models With \
         Very Large Vocabularies <https://arxiv.org/abs/1511.06909>`_

    .. seealso:: :class:`~chainer.links.BlackOut`.

    """

    batch_size = x.shape[0]
    pn = F.concat([t[:, None], samples], axis=1)
    pn_emb = embed_id.embed_id(pn, W, ignore_label=-1)
    # (N, S+1, units)
    pn_y = F.batch_matmul(pn_emb, x)
    pn_y = reshape.reshape(pn_y, pn_y.shape[:-1])
    pn_log_q = log_q[pn.data]
    pn_ly = pn_y + pn_log_q
    xp = cuda.get_array_module(t)
    mask = xp.zeros(pn_ly.shape).astype('f')
    mask[:, 1:] -= (samples.data == -1) * 1024.
    pn_ly += mask

    last = xp.zeros((batch_size, )).astype('i')
    loss = F.softmax_cross_entropy(pn_ly, last, reduce=reduce)
    return loss


class BlackOut(link.Link):

    """BlackOut loss layer.
    .. seealso:: :func:`~chainer.functions.black_out` for more detail.
    Args:
        in_size (int): Dimension of input vectors.
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.
    Attributes:
        W (~chainer.Parameter): Weight parameter matrix.
    """

    def __init__(self, in_size, counts, sample_size):
        super(BlackOut, self).__init__()
        vocab_size = len(counts)
        p = numpy.array(counts, dtype=numpy.float32)
        self.sampler = walker_alias.WalkerAlias(p)
        self.sample_size = sample_size
        self.log_q = - np.log(p + 1e-8) 
        with self.init_scope():
            self.W = variable.Parameter(shape=(vocab_size, in_size))
            
    def to_cpu(self):
        super(BlackOut, self).to_cpu()
        self.sampler.to_cpu()
        self.log_q = cuda.to_cpu(self.log_q)

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(BlackOut, self).to_gpu()
            self.sampler.to_gpu()
            self.log_q = cuda.to_gpu(self.log_q)
            
    def __call__(self, x, t):
        """Computes the loss value for given input and ground truth labels.
        Args:
            x (~chainer.Variable): Input of the weight matrix multiplication.
            t (~chainer.Variable): Batch of ground truth labels.
        Returns:
            ~chainer.Variable: Loss value.
        """

        batch_size = x.shape[0]
        if hasattr(self, 'sample_data'):
            # for test
            sample_data = self.sample_data
        else:
            shape = (batch_size, self.sample_size)
            sample_data = self.sampler.sample(shape)

        # remove t from samples explicitly, setting -1
        if isinstance(t, chainer.Variable):
            t_data = t.array
        else:
            t_data = t
        is_not_fake_correct = (sample_data != t_data[:, None])
        sample_data = self.xp.where(
            is_not_fake_correct, sample_data,
            - self.xp.ones(sample_data.shape)).astype(np.int32)

        samples = variable.Variable(sample_data)
        return black_out(x, t, self.W, self.log_q, samples)
