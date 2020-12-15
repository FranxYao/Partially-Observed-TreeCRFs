"""Tree CRF for partially observed nested NER"""

import torch
import torch_model_utils as tmu
from torch.distributions import Bernoulli
from torch_struct import TreeCRF
from tree_crf_layer import TreeCRFLayer
from tree_crf import TreeCRF_

log_potentials = torch.Tensor([[[[2, 1], [2, 1], [4, 1]],
                                [[0, 0], [4, 2], [3, 1]],
                                [[0, 0], [0, 0], [4, 3]]]])
mask = torch.Tensor([[[[0, 1], [0, 0], [0, 1]],
                      [[0, 0], [0, 1], [0, 1]],
                      [[0, 0], [0, 0], [0, 1]]]])
mask = torch.Tensor([[[[0, 0, 1], [0, 0, 0], [0, 1, 0]],
                      [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[0, 0, 0], [0, 0, 0], [0, 1, 0]]]])
lengths = torch.Tensor([3]).long()

layer = TreeCRFLayer()
log_prob_0, ent = layer(log_potentials, mask, lengths)
outputs = layer.decode_argmax(log_potentials, lengths)

log_potentials = torch.randn(2, 10, 10, 20)
lengths = torch.Tensor([5, 10]).long()
outputs = layer.decode_argmax(log_potentials, lengths)

lengths = torch.Tensor([5, 10]).long()
log_potentials = torch.randn(2, 10, 10, 20)
dist = TreeCRF(log_potentials, lengths)
dist.partition

dist_ = TreeCRF_(log_potentials, lengths)
log_z = dist_.partition
before_lse = tmu.batch_index_select(beta, lengths - 1)
before_lse = tmu.batch_index_select(before_lse, lengths - 1)

random_partial_tree = Bernoulli(0.5).sample(log_potentials.size())

tree = random_partial_tree
masked_potential = log_potentials - 1000000 * (1 - tree)
dist_mask = TreeCRF(masked_potential)
dist_mask.partition
assert (dist.log_prob(max_tree) + dist.partition == dist_.partition)
assert (dist_.log_prob(max_tree) == 0.)

lengths = torch.tensor([10, 10]).long()
label_size = 20
ratio = 0.95
mask_smooth = get_structure_smoothing_mask(tree, lengths, ratio, label_size)
smoothed_potential = log_potentials + mask_smooth.log()

dist_smooth = TreeCRF(smoothed_potential)
dist_smooth.partition

## test structure smoothing

mask = torch.Tensor([[
    [[1, 1], [1, 0], [0, 1]],
    [[0, 0], [1, 1], [1, 1]],
    [[0, 0], [0, 0], [1, 1]]
],
    [
        [[1, 1], [1, 1], [0, 1]],
        [[0, 0], [1, 1], [1, 0]],
        [[0, 0], [0, 0], [1, 1]]
    ]])
lengths = torch.Tensor([3, 3])
batch_size = 2
ratio = 0.95
mask_smooth = get_structure_smoothing_mask(mask, lengths, ratio, batch_size, label_size)
