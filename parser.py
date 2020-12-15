import torch

from copy import deepcopy
from torch import nn


class Bilinear(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.parse_proj = nn.Parameter(
            torch.randn(config.label_size, config.hidden_size, config.hidden_size))
        return

    def forward(self, sent_states):
        # prepare for tree CRF
        label_size = self.parse_proj.size(0)
        batch_size = sent_states.size(0)
        max_len = sent_states.size(1)
        hidden_size = sent_states.size(2)
        sent_states = sent_states.view(batch_size, 1, max_len, hidden_size)
        sent_states_ = sent_states.transpose(2, 3)  # [batch, 1, hidden_size, max_len]
        parse_proj = self.parse_proj.view(1, label_size, hidden_size, hidden_size)

        # project to CRF potentials
        # [batch, 1, len, hidden] * [1, label, hidden, hidden] -> [batch, label, len, hidden]
        proj = torch.matmul(sent_states, parse_proj)
        # [batch, label, len, hidden] * [batch, 1, hidden, len] -> [batch, label, len, len]
        log_potentials = torch.matmul(proj, sent_states_)
        # [batch, label, len, len] -> [batch, label, len * len] -> [[batch, len * len, label]
        log_potentials = log_potentials.view(batch_size, label_size, -1).transpose(1, 2)
        # [[batch, len * len, label] -> [[batch, len, len, label]
        log_potentials = log_potentials.view(batch_size, max_len, max_len, label_size)
        return log_potentials


class BiAffine(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.parse_proj = nn.Parameter(
            torch.randn(config.label_size, config.hidden_size, config.hidden_size))
        self.offset_proj = nn.Parameter(
            torch.randn(config.hidden_size, config.label_size))
        self.offset = nn.Parameter(torch.randn(config.label_size))
        return

    def forward(self, sent_states):
        label_size = self.parse_proj.size(0)
        batch_size = sent_states.size(0)
        max_len = sent_states.size(1)
        hidden_size = sent_states.size(2)
        sent_states = sent_states.view(batch_size, 1, max_len, hidden_size)
        sent_states_ = sent_states.transpose(2, 3)  # [batch, 1, hidden_size, max_len]
        parse_proj = self.parse_proj.view(1, label_size, hidden_size, hidden_size)

        # project to CRF potentials

        # binear part
        # [batch, 1, len, hidden] * [1, label, hidden, hidden] -> [batch, label, len, hidden]
        proj = torch.matmul(sent_states, parse_proj)
        # [batch, label, len, hidden] * [batch, 1, hidden, len] -> [batch, label, len, len]
        log_potentials = torch.matmul(proj, sent_states_)
        # [batch, label, len, len] -> [batch, label, len * len] -> [[batch, len * len, label]
        log_potentials = log_potentials.view(batch_size, label_size, -1).transpose(1, 2)
        # [[batch, len * len, label] -> [[batch, len, len, label]
        log_potentials_0 = log_potentials.view(batch_size, max_len, max_len, label_size)

        # local offset
        sent_states_sum_0 = sent_states.view(batch_size, max_len, 1, hidden_size)
        sent_states_sum_1 = sent_states.view(batch_size, 1, max_len, hidden_size)
        # [batch, len, 1, hidden] + [batch, 1, len, hidden] -> [batch, len, len, hidden]
        sent_states_sum = (sent_states_sum_0 + sent_states_sum_1).view(batch_size, -1, hidden_size)
        offset_proj = self.offset_proj.view([1, hidden_size, -1])
        # [batch, len * len, hidden] * [1, hidden, label] -> [batch, len * len, label]
        log_potentials_1 = torch.matmul(sent_states_sum, offset_proj)
        log_potentials_1 = log_potentials_1.view(batch_size, max_len, max_len, label_size)

        offset = self.offset.view(1, 1, 1, label_size)
        log_potentials = log_potentials_0 + log_potentials_1 + offset
        return log_potentials


class DeepBiaffine(nn.Module):
    def __init__(self, config):
        super().__init__()

        config_ = deepcopy(config)
        config_.hidden_size = config.hidden_size // 2
        self.biaffine = BiAffine(config_)
        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config_.hidden_size),
            nn.Dropout(config.parser_dropout),
            nn.Linear(config_.hidden_size, config_.hidden_size),
            nn.Dropout(config.parser_dropout)
        )
        return

    def forward(self, sent_states):
        sent_states = self.linear(sent_states)
        log_potentials = self.biaffine(sent_states)
        return log_potentials
