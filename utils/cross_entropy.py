# https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def _is_long(x):
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(logits, target, weight=None, ignore_index=-100, reduction='mean'):
    """cross entropy loss with support for target distributions"""

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target):
        return F.cross_entropy(logits, target, weight, ignore_index=ignore_index, reduction=reduction)

    masked_indices = None
    num_classes = logits.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    # log-softmax of logits
    lsm = F.log_softmax(logits, dim=-1)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input, target):
        return cross_entropy(input, target, self.weight, self.ignore_index, self.reduction)
