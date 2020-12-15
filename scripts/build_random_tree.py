import torch 
from .. import torch_model_utils as tmu 
from torch_struct import TreeCRF
import matplotlib.pyplot as plt


def build_compare_trees(lengths, max_len):
    """Build trees to compare

    Args:
      lengths: size=[batch], type=torch.LongTensor
      max_len: type=int

    Returns:
      random_trees: size=[batch, max_len, max_len], type=torch.LongTensor
      left_trees: size=[batch, max_len, max_len], type=torch.LongTensor
      right_trees: size=[batch, max_len, max_len], type=torch.LongTensor
    """
    batch_size = lengths.size(0)
    device = lengths.device

    log_potentials = torch.zeros(batch_size, max_len, max_len, 1).to(device)
    random_trees = TreeCRF(log_potentials, lengths).sample([1]).squeeze()

    mask = tmu.lengths_to_squared_mask(lengths, max_len)

    left_trees = torch.eye(max_len).unsqueeze(0).to(device)
    left_trees = left_trees.repeat(batch_size, 1, 1)
    left_trees[:, 0] = 1
    left_trees = (left_trees * mask).long()

    diag = torch.eye(max_len).unsqueeze(0).to(device)
    diag = diag.repeat(batch_size, 1, 1)
    right_trees = torch.zeros(batch_size, max_len, max_len).to(device)
    right_trees = tmu.batch_index_fill(right_trees, lengths - 1, 1)
    right_trees = right_trees.transpose(1, 2)
    right_trees = (right_trees + diag) * mask
    right_trees = (right_trees > 0).long()
    return random_trees -1 , left_trees - 1 , right_trees -1


def test_build_tree():
    lengths = torch.Tensor([3, 4]).long()
    max_len = 5

    random_trees, left_trees, right_trees = build_compare_trees(lengths, max_len)
    batch_size = 2

    trees_ = random_trees.transpose(1, 0).reshape(max_len, batch_size * max_len)
    plt.imshow(trees_)
    plt.show()

    trees_ = left_trees.transpose(1, 0).reshape(max_len, batch_size * max_len)
    plt.imshow(trees_)
    plt.show()

    trees_ = right_trees.transpose(1, 0).reshape(max_len, batch_size * max_len)
    plt.imshow(trees_)
    plt.show()


def generate_tree():
    f = open('/Users/jamie/Desktop/MyFiles/2020/cfg/data/ACE2005/test.data')
    w = open('/Users/jamie/Desktop/tree.txt', 'w')
    lines = [x.strip() for x in f.readlines()]
    for idx in range(0, len(lines), 4):
        l = len(lines[idx].split(' '))
        lengths = torch.Tensor([l]).long()
        max_len = l
        random_trees, left_trees, right_trees = build_compare_trees(lengths, max_len)
        random_tree_str = list()
        left_tree_str = list()
        right_tree_str = list()
        for i in range(l):
            for j in range(l):
                if random_trees[i][j] >= 0:
                    random_tree_str.append("{}_{}_{}".format(i, j, int(random_trees[i][j])))
                if left_trees[0][i][j] >= 0:
                    left_tree_str.append("{}_{}_{}".format(i, j, int(left_trees[0][i][j])))
                if right_trees[0][i][j] >= 0:
                    right_tree_str.append("{}_{}_{}".format(i, j, int(right_trees[0][i][j])))
        w.write(" ".join(random_tree_str) + "\n")
        w.write(" ".join(left_tree_str) + "\n")
        w.write(" ".join(right_tree_str) + "\n")
    f.close()
    w.close()


generate_tree()
