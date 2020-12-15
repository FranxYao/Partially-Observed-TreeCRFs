"""Tree CRF Layer for partially observed nested NER"""

import torch

from torch.distributions import Categorical
from torch import nn
from torch_struct import TreeCRF
from tree_crf import TreeCRFVanilla
import torch_model_utils as tmu


def get_structure_smoothing_mask(mask, lengths, ratio):
    """
    Args:
      mask:
      lengths:
      ratio: Float,
      label_size: Int,

    Returns:
      mask_smooth:
    """
    inspect = {}

    batch_size = mask.size(0)
    label_size = mask.size(-1)
    mask_observed = mask * ratio
    mask_observed_sum = mask_observed.view(batch_size, -1).sum(-1)
    total_observed = mask.view(batch_size, -1).sum(-1)
    smooth_budget = total_observed - mask_observed_sum
    total_location = label_size * lengths * (lengths + 1) / 2
    smooth_p = smooth_budget.float() / (total_location - total_observed)
    # print('DEBUG, smooth_p.size = ', smooth_p.size())
    if (smooth_p.mean() > ratio):
        raise ValueError(
            'Smoothing ratio too strong, observed : reject = %.2f: %.2f'
            % (ratio, smooth_p))
    smooth_p = smooth_p.view(batch_size, 1, 1, 1)
    mask_rejected = (1 - mask) * smooth_p
    mask_smooth = mask_observed + mask_rejected
    return mask_smooth


def get_structure_smoothing_mask_v1(mask, lengths, ratio):
    """
    Args:
      mask:
      lengths:
      ratio: Float,
      label_size: Int,

    Returns:
      mask_smooth:
    """
    # print('DEBUG, mask.size = ', mask.size())
    mask_observed = mask
    mask_rejected = (1 - mask) * (1 - ratio)
    mask_smooth = mask_observed + mask_rejected
    return mask_smooth


class TreeCRFLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.potential_normalization = config.potential_normalization
        self.observed_label_size = config.observed_label_size
        self.dropout = config.state_dropout_p
        self.dropout_mode = config.state_dropout_mode
        self.structure_smoothing = config.structure_smoothing_p
        self.decode_method = config.decode_method
        self.use_vanilla_crf = config.use_vanilla_crf
        self.no_batchify = config.no_batchify
        return

    def forward(self, log_potentials, mask, lengths):
        """Partially marginalize the given tree

        Args:
          log_potentials: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]
          mask: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]. 1 = not masked, 0 = masked
          lengths: torch.LongTensor, size=[batch]

        Returns:
          log_prob: torch.FloatTensor, size=[batch]
          entropy: torch.FloatTensor, size=[batch]
        """
        inspect = {}

        device = log_potentials.device
        batch_size = log_potentials.size(0)
        max_len = log_potentials.size(1)
        label_size = mask.size(-1)

        if(self.use_vanilla_crf): TreeCRF_ = TreeCRFVanilla
        else: TreeCRF_ = TreeCRF

        if (self.potential_normalization):
            lp_mean = log_potentials.reshape(batch_size, -1).mean(-1)
            lp_std = log_potentials.reshape(batch_size, -1).std(-1)
            log_potentials = log_potentials - lp_mean.view(batch_size, 1, 1, 1)
            log_potentials = log_potentials / lp_std.view(batch_size, 1, 1, 1)

        full_crf = TreeCRF_(log_potentials, lengths=lengths)
        z_full = full_crf.partition
        inspect['z_full'] = z_full.mean().item()
        entropy = full_crf.entropy

        # State dropout
        if (self.dropout > 0.0):
            dropout_dist = Categorical(torch.tensor([self.dropout, 1. - self.dropout]))

            # observed mask
            # [batch, max_len, max_len]
            dropout_mask_observed = dropout_dist.sample(mask.size()[:-1]).to(device)
            # [batch, max_len, max_len, observed_label_size]
            dropout_mask_observed = dropout_mask_observed.unsqueeze(-1) \
                .repeat(1, 1, 1, self.observed_label_size)
            ones_latent_ = torch.ones(mask.size())[:, :, :, self.observed_label_size:]
            ones_latent_ = ones_latent_.long()
            # [batch, max_len, max_len, latent_label_size]
            ones_latent_ = ones_latent_.to(device)
            dropout_mask_observed = torch.cat(
                [dropout_mask_observed, ones_latent_], dim=3)

            # latent mask
            dropout_mask_latent = dropout_dist.sample(mask.size()).to(device)
            if (self.dropout_mode == 'full'):
                mask *= dropout_mask_observed
                dropout_mask_latent[:, :, :, :self.observed_label_size] = 1.
                mask *= dropout_mask_latent
            elif (self.dropout_mode == 'latent'):
                dropout_mask_latent[:, :, :, :self.observed_label_size] = 1.
                mask *= dropout_mask_latent
            else:
                raise NotImplementedError('Illegal dropout mode %s' % self.dropout_mode)

        # Structure smoothing
        if (self.structure_smoothing < 1.0):
            if (self.dropout > 0.0):
                raise ValueError('do not support state dropout when doing smoothing!')
            mask_smooth = get_structure_smoothing_mask_v1(
                mask, lengths, self.structure_smoothing)
            smoothed_potentials = log_potentials + torch.log(mask_smooth + 1e-10)
            smoothed_crf = TreeCRF_(smoothed_potentials, lengths=lengths)
            z_smooth = smoothed_crf.partition
            log_prob_smooth = z_smooth - z_full
            inspect['z_smooth'] = z_smooth.mean().item()
        else:
            log_prob_smooth = torch.zeros(batch_size) - 1

        masked_potentials = log_potentials - 1000000 * (1 - mask)
        if(self.no_batchify):
          z_partial = []
          for i in range(batch_size):
            potential_i = masked_potentials[i].unsqueeze(0)
            len_i = lengths[i].unsqueeze(0)
            z_partial.append(TreeCRF_(potential_i, len_i).partition[0])
          z_partial = torch.stack(z_partial)
        else:
          masked_crf = TreeCRF_(masked_potentials, lengths=lengths)
          z_partial = masked_crf.partition
        inspect['z_partial'] = z_partial.mean().item()

        log_prob = z_partial - z_full
        return log_prob, log_prob_smooth, entropy, inspect

    def decode(self, log_potentials, lengths):
        """Decode the max-prob tree

        Args:
          log_potentials: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]
          mask: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]. 1 = not masked, 0 = masked

        Returns:
          trees: torch.LongTensor, size=[batch, max_len, max_len]
            trees[bi, j, k] = l means for the sentence bi in a batch, there is a
            constituent labeled l (l != 0) ranging from location j to
        """
        label_size = log_potentials.size(-1)
        device = log_potentials.device

        if(self.decode_method == 'argmax'):
          crf = TreeCRF(log_potentials, lengths=lengths)
          trees = crf.argmax
        elif(self.decode_method == 'marginal'):
          crf = TreeCRF(log_potentials, lengths=lengths)
          marginals = crf.marginals
          crf_marginal = TreeCRF((marginals + 1e-10).log(), lengths=lengths)
          trees = crf_marginal.argmax
        else: 
          raise NotImplementedError(
            'decode method %s not implemented' % self.decode_method)

        ind = 1 + torch.arange(label_size).to(device).view(1, 1, 1, -1)
        trees = (trees * ind).sum(dim=-1)
        trees = trees - 1
        return trees
