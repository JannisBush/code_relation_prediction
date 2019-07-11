# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2019 Jannis Rautenstrauch
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np
import pandas as pd

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss

from tensorboardX import SummaryWriter

from sklearn.model_selection import StratifiedKFold

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from run_classifier_dataset_utils import processors, convert_examples_to_features, compute_metrics

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


logger = logging.getLogger(__name__)


def main():
    """Fine-tune BERT for a given task with given parameters."""

    # Define all parameters, using argparse/Command Line Interface
    parser = argparse.ArgumentParser()

    def add_args():
        """Add all possible options and defaults to the parser."""
        parser.add_argument("--data_dir",
                            default="../data",
                            type=str,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        parser.add_argument("--bert_model",
                            default="bert-base-uncased",
                            type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                            "bert-base-multilingual-cased, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default="node",
                            type=str,
                            help="The name of the task to train. One of node, political-as, "
                                 "political-ru, political-asu, agreement")
        parser.add_argument("--output_dir",
                            default="run",
                            type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--cache_dir",
                            default="",
                            type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")
        parser.add_argument("--input_to_use",
                            type=str,
                            default="both",
                            help="Which input to use. One of both, org, response.")
        parser.add_argument("--do_train",
                            action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval",
                            action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--do_cross_val",
                            action='store_true',
                            help="Whether to run cross-validation.")
        parser.add_argument("--do_save",
                            action='store_true',
                            help="Whether to save the resulting model.")
        parser.add_argument("--do_visualization",
                            action='store_true',
                            help="Whether to run visualization.")
        parser.add_argument("--do_lower_case",
                            action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--train_batch_size",
                            default=16,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=8,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=2e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=3.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--no_cuda",
                            action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument('--overwrite_output_dir',
                            action='store_true',
                            help="Overwrite the content of the output directory")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--log_level',
                            type=str,
                            default="info",
                            help="Verbosity of logging output. One of info or warn.")

    add_args()
    args = parser.parse_args()

    # Set up all parameters given the CLI arguments
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device = device
    task_name = args.task_name.lower()
    processor = processors[task_name](args.input_to_use)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    global_step = 0
    tr_loss = 0
    tb_writer = SummaryWriter()

    # Prepare the logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.log_level == "info" else logging.WARN)
    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    # Fail if the arguments are invalid
    if not args.do_train and not args.do_eval and not args.do_cross_val and not args.do_visualization:
        raise ValueError("At least one of `do_train`, `do_eval` `do_cross_val` or `do_visualization` must be True.")
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. "
                         "Use the --overwrite_output_dir option.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Set all seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    def get_features_examples(mode):
        """Returns the features and examples of train or test mode."""

        def convert(split, modus, exs):
            """Converts the examples or load them from cache."""
            cached_features_file = os.path.join(args.data_dir, '{0}_{1}_{2}_{3}_{4}_{5}'.format(split, modus,
                list(filter(None, args.bert_model.split('/'))).pop(),
                            str(args.max_seq_length),
                            str(task_name), str(args.input_to_use)))
            try:
                with open(cached_features_file, "rb") as reader:
                    fs = pickle.load(reader)
            except:
                fs = convert_examples_to_features(
                    exs, label_list, args.max_seq_length, tokenizer)

                logger.info('Saving {0} features into cached file {1}'.format(mode, cached_features_file))
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(fs, writer)

            return fs

        # Prepare data loader
        if mode == "train":
            train_ex, df = processor.get_train_examples(args.data_dir)
            return convert("X", mode, train_ex), train_ex, df
        elif mode == "dev":
            dev_ex, df = processor.get_dev_examples(args.data_dir)
            return convert("X", mode, dev_ex), dev_ex, df
        elif mode == "cross_val":
            data = processor.get_splits(args.data_dir)
            train_f_list, train_e_list, train_df_list, test_f_list, test_e_list, test_df_list = ([] for _ in range(6))
            for i, (train_ex, train_df, test_ex, test_df) in enumerate(data):
                train_e_list.append(train_ex)
                train_df_list.append(train_df)
                test_e_list.append(test_ex)
                test_df_list.append(test_df)
                # Create features
                train_f_list.append(convert(i, "train", train_ex))
                test_f_list.append(convert(i, "dev", test_ex))
            return train_f_list, train_e_list, train_df_list, test_f_list, test_e_list, test_df_list
        else:
            raise ValueError("Invalid feature mode.")

    def do_training(train_features, train_examples):
        """Runs BERT fine-tuning."""
        # Allows to write to enclosed variables tokenizer, model and global_step
        nonlocal global_step

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                loss = CrossEntropyLoss()(logits.view(-1, num_labels), label_ids.view(-1))

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', loss.item(), global_step)

    def do_save():
        """Saves the current model, tokenizer and arguments."""
        nonlocal model
        nonlocal tokenizer
        ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        ### Example:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)

    def do_eval(eval_features, eval_examples):
        """Do evaluation on the current model."""
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)

        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        out_label_ids = None

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            # create eval loss and other metric required by the task
            tmp_eval_loss = CrossEntropyLoss()(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        preds = np.argmax(preds, axis=1)

        result = compute_metrics(task_name, preds, out_label_ids)

        loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss
        # Save all settings for external evaluation
        result['_task'] = task_name
        result['_input_mode'] = args.input_to_use
        result['_learning_rate'] = args.learning_rate
        result['_bert-model'] = args.bert_model
        result['_batch_size'] = args.train_batch_size
        result['_warmup'] = args.warmup_proportion
        result['_num_epochs'] = args.num_train_epochs
        result['_seq_len'] = args.max_seq_length
        result['_seed'] = args.seed
        result['_gradient_acc'] = args.gradient_accumulation_steps

        return result, preds

    def save_results(result_list, pred_list):
        """Saves the results."""
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for i, result_dict in enumerate(result_list):
                logger.info("Run %i", i)
                writer.write("Run %i\n" % i)
                for key in sorted(result_dict.keys()):
                    if not key.startswith("_"):
                        logger.info("  %s = %s", key, str(result_dict[key]))
                        writer.write("%s = %s\n" % (key, str(result_dict[key])))
        output_csv_file = os.path.join(args.output_dir, "eval_results.csv")
        output_preds_file = os.path.join(args.output_dir, "eval_preds.csv")
        df_res = pd.DataFrame(result_list)
        df_preds = pd.DataFrame(pred_list)
        df_res.to_csv(output_csv_file, encoding='utf-8', sep='\t', index=False)
        df_preds.to_csv(output_preds_file, encoding='utf-8', index=False)

    # Load the tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    # Training
    if args.do_train:
        features, examples, df = get_features_examples("train")
        do_training(features, examples)
        if args.do_save:
            do_save()

    # Evaluation
    if args.do_eval:
        features, examples, df = get_features_examples("dev")
        result, preds = do_eval(features, examples)
        # Analyze results, somewhere else,
        # the corresponding test_dfs can be loaded using get_features_examples("dev")
        # print(preds)
        # df['predictions'] = preds
        # df['correctness'] = df.apply(lambda r: 1 if r['label'] == r['predictions'] else 0, axis=1)
        # rels = pd.crosstab(df['topic'], [df['label'], df['predictions']], margins=True,
        #                    colnames=['label', 'prediction'])
        # rels2 = pd.crosstab(df['topic'], df['correctness'], normalize='index')
        # print(rels)
        # print(rels2)
        save_results([result], [preds])

    # CrossVal
    if args.do_cross_val:
        results = []
        predictions = []
        train_f_l, train_e_l, train_df_l, test_f_l, test_e_l, test_df_l = get_features_examples("cross_val")
        for train_features, train_examples, test_features, test_examples in zip(
                train_f_l, train_e_l, test_f_l, test_e_l):
            # Reset model
            model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
            model.to(device)
            # Train
            do_training(train_features, train_examples)
            # Eval
            result, preds = do_eval(test_features, test_examples)
            results.append(result)
            predictions.append(preds)

        # Analyze results, somewhere else,
        # the corresponding test_dfs can be loaded using get_features_examples("cross_val")
        #result_df = pd.DataFrame(results)
        #print(result_df.agg([np.mean, np.max, np.min]))

        save_results(results, predictions)

    # Visualization
    if args.do_visualization:
        from skorch import NeuralNetClassifier
        visu_features, _, _ = get_features_examples("dev")

        all_input_ids = torch.tensor([f.input_ids for f in visu_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in visu_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in visu_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in visu_features], dtype=torch.long)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.model = model

            def forward(self, X):
                return self.model(*X)

        net = NeuralNetClassifier(MyModule, max_epochs=0, lr=0.0, device='cuda')
        net.fit([all_input_ids, all_segment_ids, all_input_mask], y=all_label_ids)
        y_proba = net.predict_proba([all_input_ids, all_segment_ids, all_input_mask])
        print(preds)
        print(net.predict([all_input_ids, all_segment_ids, all_input_mask]))


if __name__ == "__main__":
    main()
