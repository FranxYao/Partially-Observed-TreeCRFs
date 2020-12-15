import torch


def batch_index_select(A, ind):
    """Batched index select

    Args:
      A: size=[batch, num_class, *]
      ind: size=[batch, num_select] or [batch]

    Returns:
      A_selected: size=[batch, num_select, *] or [batch, *]
    """
    batch_size = A.size(0)
    num_class = A.size(1)
    A_size = list(A.size())
    device = A.device
    A_ = A.clone().reshape(batch_size * num_class, -1)
    if (len(ind.size()) == 1):
        batch_ind = (torch.arange(batch_size) * num_class) \
            .type(torch.long).to(device)
        ind_ = ind + batch_ind
        A_selected = torch.index_select(A_, 0, ind_) \
            .view([batch_size] + A_size[2:])
    else:
        batch_ind = (torch.arange(batch_size) * num_class) \
            .type(torch.long).to(device)
        num_select = ind.size(1)
        batch_ind = batch_ind.view(batch_size, 1)
        ind_ = (ind + batch_ind).view(batch_size * num_select)
        A_selected = torch.index_select(A_, 0, ind_) \
            .view([batch_size, num_select] + A_size[2:])
    return A_selected


def length_to_mask(length, max_len):
    """
    True = 1 = not masked, False = 0 = masked

    Args:
      length: type=torch.tensor(int), size=[batch]
      max_len: type=int

    Returns:
      mask: type=torch.tensor(bool), size=[batch, max_len]
    """
    batch_size = length.shape[0]
    device = length.device
    mask = torch.arange(max_len, dtype=length.dtype) \
               .expand(batch_size, max_len).to(device) < length.unsqueeze(1)
    return mask


def lengths_to_squared_mask(lengths, max_len):
    """
      True = 1 = not masked, False = 0 = masked

      e.g., lengths = [2], max_len = 3
      returns: [[1, 1, 0],
                [1, 1, 0],
                [0, 0, 0]]

      Args:
        length: type=torch.tensor(int), size=[batch]
        max_len: type=int

      Returns:
        mask: type=torch.tensor(bool), size=[batch, max_len, max_len]
      """
    batch_size = lengths.size(0)
    mask_ = length_to_mask(lengths, max_len)
    mask = mask_.view(batch_size, 1, max_len).repeat(1, max_len, 1)
    mask = mask * mask_.float().unsqueeze(-1)
    return mask.bool()


def batch_index_put(A, ind, N):
    """distribute a batch of values to given locations

    Example:
      A = tensor([[0.1000, 0.9000],
                  [0.2000, 0.8000]])
      ind = tensor([[1, 2],
                    [0, 3]])
      N = 5
    then:
      A_put = tensor([[0.0000, 0.1000, 0.9000, 0.0000, 0.0000],
                      [0.2000, 0.0000, 0.0000, 0.8000, 0.0000]])

    Args:
      A: size=[batch, M, *], * can be any list of dimensions
      ind: size=[batch, M]
      N: type=int

    Returns:
      A_put: size=[batch, N, *]
    """
    batch_size = A.size(0)
    M = A.size(1)
    As = list(A.size()[2:])
    device = A.device
    A_put = torch.zeros([batch_size * N] + As).to(device)
    ind_ = torch.arange(batch_size).view(batch_size, 1) * N
    ind_ = ind_.expand(batch_size, M).flatten().to(device)
    ind_ += ind.flatten()
    A_put[ind_] += A.view([batch_size * M] + As)
    A_put = A_put.view([batch_size, N] + As)
    return A_put


def batch_index_fill(A, ind, v):
    """Fill in values to a tensor

    Example:
      A = torch.zeros(2, 4, 2)
      ind = torch.LongTensor([[2, 3], [0, 1]])
      batch_index_fill(A, ind, 4) returns:
        tensor([[[0., 0.],
                [0., 0.],
                [4., 4.],
                [4., 4.]],

               [[4., 4.],
                [4., 4.],
                [0., 0.],
                [0., 0.]]])

    Args:
      A: size=[batch, M, rest]
      ind: size=[batch] or [batch, N], N < M
      v: size=[rest] or 1

    Returns:
      A_filled: size=[batch, M, rest]
    """
    A = A.clone()
    batch_size = A.size(0)
    M = A.size(1)
    As = list(A.size()[2:])

    A_ = A.view([batch_size * M] + As)

    if (len(ind.size()) == 1): ind = ind.unsqueeze(1)
    ind_ = (((torch.arange(batch_size)) * M).unsqueeze(1) + ind).flatten()
    A_[ind_] = v
    A_filled = A_.view([batch_size, M] + As)
    return A_filled
