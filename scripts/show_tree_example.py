import torch
from torch_struct import TreeCRF

import matplotlib.pyplot as plt

log_potentials = torch.randn(2, 10, 10, 20)
dist = TreeCRF(log_potentials)
max_tree = dist.argmax
label_size = 20
ind = 1 + torch.arange(label_size).view(1, 1, 1, -1)
trees = (max_tree * ind).sum(dim=-1)
batch_size = 2
max_len = 10

# trees.size = [batch, len, len]
trees_ = trees.transpose(1, 0).reshape(max_len, batch_size * max_len)
plt.imshow(trees_)
plt.show()
