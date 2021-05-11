# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForTokenClassification, BertTokenizer)

from utils_ner import Processor, eval

from model import PartialPCFG

import time


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--train_file", default=None, type=str, required=True,
                    help="training data")
parser.add_argument("--predict_file", default=None, type=str, required=True,
                    help="development data")
parser.add_argument("--test_file", default=None, type=str, required=True,
                    help="test data")
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="Model type -- BERT")
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--lambda_ent", default=1e-2, type=float,
                    help="lambda_ent")
parser.add_argument("--dataset", default="ACE05", type=str,
                    help="ACE04,ACE05,GENIA,NNE")
parser.add_argument("--latent_size", default=7, type=int,
                    help="latent_size")

## Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=384, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_predict", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=10, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=50,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", action='store_true',
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument('--parser_type', type=str, default='bilinear',
                    help="bilinear, biaffine, or deepbiaffine")
parser.add_argument('--parser_dropout', type=float, default=0.,
                    help="parser dropout probability")
parser.add_argument('--state_dropout_p', type=float, default=0.0,
                    help="state dropout probability")
parser.add_argument('--state_dropout_mode', type=str, default='latent',
                    help="state dropout mode, latent or full")
parser.add_argument('--structure_smoothing_p', type=float, default=1.0,
                    help="structure smoothing ratio")
parser.add_argument('--potential_normalization', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if use potential normalization")
parser.add_argument('--use_vanilla_crf', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if use vanilla crf implementation, slow speed")
parser.add_argument('--use_crf', type=str2bool,
                    nargs='?', const=True, default=True,
                    help="if use crf for parsing, else use local normalization")
parser.add_argument('--decode_method', type=str, default='argmax',
                    help="argmax or marginal")
parser.add_argument('--no_batchify', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if do not batchify the inside algorithm")
parser.add_argument('--full_print', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if full print for tree, else only print not latent node")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--gpu_id', type=int, default=-1, help="gpu_id")

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
log_file_name = os.path.join(args.output_dir, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer)
}

best_F = 0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer, processor):
    """ Train the model """

    global best_F

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        loss_ = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'gather_ids': batch[3],
                      'gather_masks': batch[4],
                      'partial_masks': batch[5],
                      }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            loss_ += loss.item()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                # optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, dev_dataset, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model,
                #                                             'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #     best_F = evaluate(args, model, dev_dataset, tokenizer, processor, global_step, _, best_F, "dev")
                #     evaluate(args, model, test_dataset, tokenizer, processor, global_step, _, best_F, "test")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # model_to_save = model.module if hasattr(model,
        #                                         'module') else model  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(output_dir)
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        # logger.info("Saving model checkpoint to %s", output_dir)

        evaluate(args, model, dev_dataset, tokenizer, processor, global_step, _, "dev")
        evaluate(args, model, test_dataset, tokenizer, processor, global_step, _, "test")

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dataset, tokenizer, processor, step, _, prefix):
    global best_F

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    nb_eval_steps, nb_eval_examples = 0, 0
    correct, precision, recall = 0, 0, 0


    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'gather_ids': batch[3],
                      'gather_masks': batch[4]
                      }
            if isinstance(model, torch.nn.DataParallel):
                outputs = model.module.infer(**inputs)
            else:
                outputs = model.infer(**inputs)
            correct_count, pred_count, gold_count = eval(args, outputs[0], batch[5], len(processor.labels),
                                                                  batch[4])
            correct += correct_count
            precision += pred_count
            recall += gold_count


        nb_eval_examples += inputs['input_ids'].size(0)
        nb_eval_steps += 1

    if precision > 0 and recall > 0 and correct > 0:
        precision = correct / precision
        recall = correct / recall
        F = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        F = 0

    result = {'prefix': prefix,
              'epoch': _,
              'eval_precision': precision,
              'eval_recall': recall,
              'eval_F': F,
              }

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        # writer.write("Epoch = %d Step = %d Total Loss = %d \n" % _, step, tr_loss)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n")

    if prefix == 'dev' and F > best_F:
        best_F = F

        output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}-{}'.format(_, step, int(F * 10000)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    model.train()

    return


def load_and_cache_examples(args, tokenizer, processor, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file

    logger.info("Creating features from dataset file at %s", args.train_file)
    train_examples = processor.get_train_examples(input_file=args.train_file)
    logger.info("Creating features from dataset file at %s", args.predict_file)
    dev_examples = processor.get_dev_examples(input_file=args.predict_file)
    logger.info("Creating features from dataset file at %s", args.test_file)
    test_examples = processor.get_dev_examples(input_file=args.test_file)

    train_features = processor.convert_examples_to_features(train_examples, args.max_seq_length,
                                                            tokenizer)
    dev_features = processor.convert_examples_to_features(dev_examples, args.max_seq_length,
                                                          tokenizer)
    test_features = processor.convert_examples_to_features(test_examples, args.max_seq_length,
                                                           tokenizer)

    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", dev_cached_features_file)
    #     torch.save(dev_features, dev_cached_features_file)

    # Convert to Tensors and build dataset

    def getTensorDataset(features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_gather_ids = torch.tensor([f.gather_ids for f in features], dtype=torch.long)
        all_gather_masks = torch.tensor([f.gather_masks for f in features], dtype=torch.long)
        all_partial_masks = torch.tensor([f.partial_masks for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_gather_ids, all_gather_masks, all_partial_masks)
        return dataset

    train_dataset = getTensorDataset(train_features)
    dev_dataset = getTensorDataset(dev_features)
    test_dataset = getTensorDataset(test_features)

    return train_dataset, dev_dataset, test_dataset


def main():
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, do_basic_tokenize=False)

    processor = Processor(logger, args.dataset, args.latent_size)

    config.label_size = len(processor.labels) + args.latent_size
    config.observed_label_size = len(processor.labels)
    config.latent_label_size = config.label_size - config.observed_label_size
    config.parser_type = args.parser_type
    config.parser_dropout = args.parser_dropout
    config.state_dropout_p = args.state_dropout_p
    config.state_dropout_mode = args.state_dropout_mode
    config.lambda_ent = args.lambda_ent
    config.structure_smoothing_p = args.structure_smoothing_p
    config.potential_normalization = args.potential_normalization
    config.decode_method = args.decode_method
    config.use_vanilla_crf = args.use_vanilla_crf
    config.use_crf = args.use_crf
    config.full_print = args.full_print
    config.no_batchify = args.no_batchify

    model = PartialPCFG.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    train_dataset, dev_dataset, test_dataset = load_and_cache_examples(args, tokenizer, processor, evaluate=False,
                                                                       output_examples=False)

    # Training

    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer, processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_predict and args.local_rank in [-1, 0]:
        checkpoint = args.output_dir
        model = PartialPCFG.from_pretrained(checkpoint, config=config)
        model.to(args.device)
        evaluate(args, model, test_dataset, tokenizer, processor, 0, 0, "eval")

    return


if __name__ == "__main__":
    main()
