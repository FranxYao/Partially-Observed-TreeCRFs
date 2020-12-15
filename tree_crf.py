import torch
import torch_model_utils as tmu

from torch import nn
from torch.distributions.utils import lazy_property


class TreeCRFVanilla(nn.Module):

    def __init__(self, log_potentials, lengths=None):
        self.log_potentials = log_potentials
        self.lengths = lengths
        return

    @lazy_property
    def entropy(self):
        batch_size = self.log_potentials.size(0)
        device = self.log_potentials.device
        return torch.zeros(batch_size).to(device)

    @lazy_property
    def partition(self):
        # Inside algorithm
        device = self.log_potentials.device
        batch_size = self.log_potentials.size(0)
        max_len = self.log_potentials.size(1)
        label_size = self.log_potentials.size(3)

        beta = torch.zeros_like(self.log_potentials).to(device)
        for i in range(max_len):
            beta[:, i, i] = self.log_potentials[:, i, i]
        for d in range(1, max_len):
            for i in range(max_len - d):
                j = i + d
                before_lse_1 = beta[:, i, i:j].view(batch_size, d, label_size, 1)
                before_lse_2 = beta[:, i + 1: j + 1, j].view(batch_size, d, 1, label_size)
                before_lse = (before_lse_1 + before_lse_2).reshape(batch_size, -1)
                after_lse = torch.logsumexp(before_lse, -1).view(batch_size, 1)
                beta[:, i, j] = self.log_potentials[:, i, j] + after_lse
        if (self.lengths is None):
            before_lse = beta[:, 0, max_len - 1]
        else:
            before_lse = tmu.batch_index_select(beta[:, 0], self.lengths - 1)
        log_z = torch.logsumexp(before_lse, -1)
        return log_z

    @lazy_property
    def argmax(self):
        raise NotImplementedError('slow argmax not implemented!')
        return
