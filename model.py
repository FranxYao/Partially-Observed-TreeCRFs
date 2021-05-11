import torch
import torch.nn.functional as F
import torch_model_utils as tmu

from transformers import BertPreTrainedModel, BertModel
from torch import nn

from tree_crf_layer import TreeCRFLayer
from parser import Bilinear, BiAffine, DeepBiaffine


def partial_mask_to_targets(mask):
    device = mask.device
    label_size = mask.size(-1)
    ind = 1 + torch.arange(label_size).to(device).view(1, 1, 1, -1)
    trees = (mask * ind).sum(dim=-1)
    trees = trees - 1
    tree_rej_ind = trees == -1
    trees[tree_rej_ind] = label_size - 1
    return trees


class PartialPCFG(BertPreTrainedModel):

    def __init__(self, config):
        super(PartialPCFG, self).__init__(config)

        self.lambda_ent = config.lambda_ent  # try [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        self.label_size = config.label_size
        self.structure_smoothing = config.structure_smoothing_p < 1.0

        self.use_crf = config.use_crf
        if (self.use_crf is False): assert (config.latent_label_size == 1)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if (config.parser_type == 'bilinear'):
            self.parser = Bilinear(config)
        elif (config.parser_type == 'biaffine'):
            self.parser = BiAffine(config)
        elif (config.parser_type == 'deepbiaffine'):
            self.parser = DeepBiaffine(config)
        else:
            raise NotImplementedError('illegal parser type %s not implemented!' % config.parser_type)
        self.tree_crf = TreeCRFLayer(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks):
        """

        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        # prepare for tree CRF
        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)
        max_len = log_potentials.size(1)
        # TODO: use vanilla span classification 
        if (self.use_crf is False):
            # [batch * max_len * max_len]
            targets = partial_mask_to_targets(partial_masks).view(-1)
            # [batch * max_len * max_len, label_size]
            prob = log_potentials.reshape(-1, label_size)
            loss = F.cross_entropy(prob, targets, reduction='none')

            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            # [batch, max_len, max_len] -> [batch * max_len * max_len]
            mask = torch.triu(mask.float()).view(-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            # log_prob_sum_partial.size = [batch]
            # TODO: check partial_masks boundary, Done
            log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
                self.tree_crf(log_potentials, partial_masks, lengths)

            if (self.structure_smoothing):
                loss = -log_prob_smooth.mean()
            else:
                loss = -log_prob_sum_partial.mean()
            loss -= self.lambda_ent * entropy.mean()

        outputs = [loss, inspect]
        return outputs

    def infer(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        """
        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
        Returns:
            outputs: list 
        """
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        if (self.use_crf is False):
            # [batch, max_len, max_len]
            trees = log_potentials.argmax(-1)
            max_len = log_potentials.size(1)
            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            mask = torch.triu(mask.float())
            trees = trees * mask - (1. - mask)
        else:
            trees = self.tree_crf.decode(log_potentials, lengths)

        outputs = [trees]
        return outputs
