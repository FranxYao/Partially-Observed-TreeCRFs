import torch
from tqdm import tqdm


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, pos, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.pos = pos
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, gather_ids, gather_masks, partial_masks):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.gather_ids = gather_ids
        self.gather_masks = gather_masks
        self.partial_masks = partial_masks


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, logger):
        self.logger = logger

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines


class Processor(DataProcessor):
    """Processor NQG data set."""

    def __init__(self, logger, dataset, latent_size):
        self.logger = logger
        if dataset == "ACE04" or dataset == "ACE05":
            self.labels = ['PER', 'LOC', 'ORG', 'GPE', 'FAC', 'VEH', 'WEA']
        elif dataset == "GENIA":
            self.labels = ['None', 'G#RNA', 'G#protein', 'G#DNA', 'G#cell_type', 'G#cell_line']
        else:
            raise NotImplementedError()

        if dataset == "ACE05" or dataset == "GENIA" or dataset == "ACE04":
            self.interval = 3
        else:
            raise NotImplementedError()

        self.latent_size = latent_size

    def get_train_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "dev")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []

        for i in range(0, len(lines), self.interval):
            text_a = lines[i]
            label = lines[i + 1]

            examples.append(
                InputExample(guid=len(examples), text_a=text_a, pos=None, label=label))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []

        for (ex_index, example) in enumerate(tqdm(examples)):

            tokens = tokenizer.tokenize(example.text_a)

            gather_ids = list()
            for (idx, token) in enumerate(tokens):
                if (not token.startswith("##") and idx < max_seq_length - 2):
                    gather_ids.append(idx + 1)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            gather_padding = [0] * (max_seq_length - len(gather_ids))
            gather_masks = [1] * len(gather_ids) + gather_padding
            gather_ids += gather_padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(gather_ids) == max_seq_length
            assert len(gather_masks) == max_seq_length

            partial_masks = self.generate_partial_masks(example.text_a.split(' '), max_seq_length, example.label,
                                                        self.labels)

            if ex_index < 2:
                self.logger.info("*** Example ***")
                self.logger.info("guid: %s" % (example.guid))
                self.logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                self.logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                self.logger.info(
                    "gather_ids: %s" % " ".join([str(x) for x in gather_ids]))
                self.logger.info(
                    "gather_masks: %s" % " ".join([str(x) for x in gather_masks]))
                # self.logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks))

        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def generate_partial_masks(self, tokens, max_seq_length, labels, tags):

        total_tags_num = len(tags) + self.latent_size

        labels = labels.split('|')
        label_list = list()

        for label in labels:
            if not label:
                continue
            sp = label.strip().split(' ')
            start, end = sp[0].split(',')[:2]
            start = int(start)
            end = int(end) - 1
            label_list.append((start, end, sp[1]))

        mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        for start, end, tag in label_list:

            if start < max_seq_length and end < max_seq_length:
                tag_idx = tags.index(tag)
                mask[start][end][tag_idx] = 1
                for k in range(total_tags_num):
                    if k != tag_idx:
                        mask[start][end][k] = 0

            for i in range(l):
                if i > end:
                    continue
                for j in range(i, l):
                    if j < start:
                        continue
                    if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
                        for k in range(total_tags_num):
                            mask[i][j][k] = 0

        for i in range(l):
            for j in range(0, i):
                for k in range(total_tags_num):
                    mask[i][j][k] = 0

        for i in range(l):
            for j in range(i, l):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        if k < len(tags):
                            mask[i][j][k] = 0
                        else:
                            mask[i][j][k] = 1

        for i in range(max_seq_length):
            for j in range(max_seq_length):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        mask[i][j][k] = 0

        return mask


class MultitasksResultItem():

    def __init__(self, id, start_prob, end_prob, span_prob, label_id, position_id, start_id, end_id):
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.span_prob = span_prob
        self.id = id
        self.label_id = label_id
        self.position_id = position_id
        self.start_id = start_id
        self.end_id = end_id


def eval(args, outputs, partial_masks, label_size, gather_masks):
    correct, pred_count, gold_count = 0, 0, 0
    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()
    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):

        golds = list()
        preds = list()

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0:
                    if output[i][j] < label_size:
                        preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

    return correct, pred_count, gold_count
