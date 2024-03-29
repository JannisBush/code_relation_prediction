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
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter

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

    # Define all parameters, using argparse/Command Line Interface.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def add_args():
        """Add all possible options and defaults to the parser."""
        # Hyperparameters of BERT
        # Parameters often changed
        parser.add_argument("--bert_model",
                            default="bert-base-uncased",
                            type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-large-cased, "
                                 "bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
        parser.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")
        parser.add_argument("--train_batch_size",
                            default=16,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--learning_rate",
                            default=2e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=3.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--do_lower_case",
                            action='store_true',
                            help="Set this flag if you are using an uncased model.")
        # Parameters usually unchanged
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--eval_batch_size",
                            default=8,
                            type=int,
                            help="Total batch size for eval.")
        # Parameters of the task
        parser.add_argument("--task_name",
                            default="node",
                            type=str,
                            help="The name of the task to train. One of node, political-as, "
                                 "political-ru, political-asu, agreement, node-ext, political-as-topics,"
                                 "political-ru-topics, political-asu-topics, agreement-topics")
        parser.add_argument("--input_to_use",
                            type=str,
                            default="both",
                            help="Which input to use. One of both, org, response, response-org.")
        # Parameters for reproduction
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        # Parameters for where to save/load data
        parser.add_argument("--data_dir",
                            default="../data",
                            type=str,
                            help="The input data dir. Should contain the .tsv file (or other data files) for the task.")
        parser.add_argument("--output_dir",
                            default="run",
                            type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--cache_dir",
                            default="",
                            type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument('--overwrite_output_dir',
                            action='store_true',
                            help="Overwrite the content of the output directory")
        # Parameters to decide what to do (train, test, crossval, save the model)
        parser.add_argument("--do_train",
                            action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval",
                            action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--do_train_eval",
                            action='store_true',
                            help="Whether to run training and eval.")
        parser.add_argument('--n_times',
                            type=int,
                            default=10,
                            help="Number of restarts for every parameter setting in train&eval mode")
        parser.add_argument("--do_cross_val",
                            action='store_true',
                            help="Whether to run cross-validation.")
        parser.add_argument("--do_save",
                            action='store_true',
                            help="Whether to save the resulting model.")
        parser.add_argument("--do_visualization",
                            action='store_true',
                            help="Whether to run visualization.")
        # Additional parameters
        parser.add_argument("--no_cuda",
                            action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument('--log_level',
                            type=str,
                            default="info",
                            help="Verbosity of logging output. One of info or warn.")

    # Add all parameters to the parser and parse them.
    add_args()
    args = parser.parse_args()

    # Set up all parameters given the CLI arguments.
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

    # Prepare the logging.
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.log_level == "info" else logging.WARN)
    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    # Check the arguments and fail if the arguments are invalid.
    if not args.do_train and not args.do_eval and not args.do_cross_val and not args.do_visualization \
            and not args.do_train_eval:
        raise ValueError("At least one of `do_train`, `do_eval` `do_cross_val` "
                         "or `do_visualization` or 'do_train_eval` must be True.")
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. "
                         "Use the --overwrite_output_dir option.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # Calculate the train_batch_size if gradient accumulation is used
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
            cached_features_file = os.path.join(args.data_dir, 'cache', '{0}_{1}_{2}_{3}_{4}_{5}'.format(modus,
                list(filter(None, args.bert_model.split('/'))).pop(),
                            str(args.max_seq_length),
                            str(task_name), str(args.input_to_use), split))
            # Try to load the cached features.
            try:
                with open(cached_features_file, "rb") as reader:
                    fs = pickle.load(reader)
            # Creates and cache the features.
            except FileNotFoundError:
                if not os.path.exists(os.path.join(args.data_dir, 'cache')):
                    os.makedirs(os.path.join(args.data_dir, 'cache'))
                fs = convert_examples_to_features(
                    exs, label_list, args.max_seq_length, tokenizer)
                logger.info('Saving {0} features into cached file {1}'.format(mode, cached_features_file))
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(fs, writer)

            return fs

        # Return the features, examples and dataframes depending on the mode.
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
                # Create features from the examples
                train_f_list.append(convert(i, "train", train_ex))
                test_f_list.append(convert(i, "dev", test_ex))
            return train_f_list, train_e_list, train_df_list, test_f_list, test_e_list, test_df_list
        else:
            raise ValueError("Invalid feature mode.")

    def create_tensor_dataset(exfeatures):
        """Creates a TensoDataset out of the features."""
        all_input_ids = torch.tensor([f.input_ids for f in exfeatures], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in exfeatures], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in exfeatures], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in exfeatures], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def do_training(train_fs, train_exs):
        """Runs BERT fine-tuning."""
        # Allows to write to enclosed variables global_step
        nonlocal global_step

        # Create the batched training data out of the features.
        train_data = create_tensor_dataset(train_fs)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Calculate the number of optimization steps.
        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer.
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

        # Log some information about the training.
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_exs))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        # Set the model to training mode and train for X epochs.
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # Iterate over all batches.
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # Get the Logits and calculate the loss.
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                loss = CrossEntropyLoss()(logits.view(-1, num_labels), label_ids.view(-1))

                # Scale the loss in gradient accumulation mode.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Calculate the gradients.
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # Update the weights every gradient_accumulation_steps steps.
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

        model_to_save = model.module if hasattr(model, 'module') else model
        # Using the predefined names, we can load using `from_pretrained`.
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        # Save the trained model, configuration and tokenizer
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Save the training arguments together with the trained model.
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)

    def do_eval(eval_features, eval_examples):
        """Do evaluation on the current model."""

        # Logg some information.
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        # Get the eval data and create a sequential dataloader.
        eval_data = create_tensor_dataset(eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Set the model to eval mode (disable dropout)
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        out_label_ids = None

        # Iterate over the evaluation data.
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            # Forward pass with deactivated autograd engine.
            with torch.no_grad():
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            # Calculate eval loss.
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

        # Calculate the mean loss and get all predictions.
        eval_loss = eval_loss / nb_eval_steps
        loss = tr_loss/global_step if args.do_train else None
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        # Compute the metrics for the given task
        result = compute_metrics(task_name, preds, out_label_ids)

        # Save additional information in the result dict.
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
        """Saves the results and the predictions."""
        # Save the results in a text file.
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            for i, result_dict in enumerate(result_list):
                logger.info("Run %i", i)
                writer.write("Run %i\n" % i)
                for key in sorted(result_dict.keys()):
                    if not key.startswith("_"):
                        logger.info("  %s = %s", key, str(result_dict[key]))
                        writer.write("%s = %s\n" % (key, str(result_dict[key])))
        # Save the results and predictions in csv and tsv files.
        output_csv_file = os.path.join(args.output_dir, "../eval_results.tsv")
        output_preds_file = os.path.join(args.output_dir, "../eval_preds.csv")
        df_res = pd.DataFrame(result_list)
        df_preds = pd.DataFrame(pred_list)
        df_preds['run'] = '{0}_{1}_{2}_{3}'.format(
            args.bert_model, args.num_train_epochs, args.train_batch_size, args.learning_rate)
        # If the files do not exist, create them with headers.
        if not os.path.exists(output_csv_file):
            df_res.to_csv(output_csv_file, encoding='utf-8', sep='\t', index=False)
            df_preds.to_csv(output_preds_file, encoding='utf-8', index=False)
        # If the files already exist, just append to them without headers.
        else:
            df_res.to_csv(output_csv_file, mode='a', encoding='utf-8', sep='\t', index=False, header=False)
            df_preds.to_csv(output_preds_file, mode='a', encoding='utf-8', index=False, header=False)

    # Load the tokenizer and the model.
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    # Train and test .
    if args.do_train_eval:
        # Get the train and test features only once.
        train_features, train_examples, _ = get_features_examples("train")
        test_features, test_examples, _ = get_features_examples("dev")

        # Repeat N times.
        for i in range(args.n_times):
            # Train.
            do_training(train_features, train_examples)
            # Eval.
            result, preds = do_eval(test_features, test_examples)
            # Save the results.
            save_results([result], [preds])
            # Reset and new seeds.
            if i+1 < args.n_times:
                args.seed += 1
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if n_gpu > 0:
                    torch.cuda.manual_seed_all(args.seed)
                # Reset model.
                model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
                model.to(device)

    # Training
    if args.do_train:
        # Get the train features.
        features, examples, df = get_features_examples("train")
        # Train.
        do_training(features, examples)
        # Save the model if wanted.
        if args.do_save:
            do_save()

    # Evaluation.
    if args.do_eval:
        # Get the dev features.
        features, examples, df = get_features_examples("dev")
        # Evaluate.
        result, preds = do_eval(features, examples)
        # Save the results.
        save_results([result], [preds])

    # CrossVal.
    if args.do_cross_val:
        # Get the data for all splits
        train_f_l, train_e_l, train_df_l, test_f_l, test_e_l, test_df_l = get_features_examples("cross_val")
        # Iterate over all splits
        for train_features, train_examples, test_features, test_examples in zip(
                train_f_l, train_e_l, test_f_l, test_e_l):
            # Reset model.
            model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
            model.to(device)
            # Train.
            do_training(train_features, train_examples)
            # Eval.
            result, preds = do_eval(test_features, test_examples)
            # Save results.
            save_results([result], [preds])

    # Visualization.
    if args.do_visualization:
        # Additional imports needed for the visualizations.
        import spacy
        from skorch import NeuralNetClassifier
        from sklearn.pipeline import make_pipeline
        from run_classifier_dataset_utils import InputExample
        from anchor import anchor_text
        from lime.lime_text import LimeTextExplainer

        # Example sentences.
        raw_text_1 = "But Mr. Nixon did n't say a word that was ever publicly recorded . Even more incredible , " \
                     "he did n't say a word when the Communists took power in Cuba - not 4 miles off their shores , " \
                     "but only 90 miles off our shores . Mr. Nixon saw what was happening in Cuba ."
        raw_text_2 = "Cordoba House is no act of tolerance, but of excess/arrogance. Building this structure on the " \
                     "edge of the battlefield created by radical Islamists is not a celebration of " \
                     "religious pluralism and mutual tolerance; it is a political statement of shocking arrogance " \
                     "and hypocrisy."
        raw_text_3 = "Are not right no does he alcohol child china play"
        raw_text_list = [raw_text_1, raw_text_2, raw_text_3]

        class BertConverter:
            """Pipeline-Class to convert text to the input format of BERT."""
            def transform(self, X, y=None, **fit_params):
                """Transforms a list of strings to a list of BERT inputs."""
                exs = []
                for text in X:
                    exs.append(InputExample(guid=None, text_a=text, text_b=None, label="attack"))
                visu_features = convert_examples_to_features(exs, label_list, args.max_seq_length, tokenizer)
                all_input_ids = torch.tensor([f.input_ids for f in visu_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in visu_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in visu_features], dtype=torch.long)
                return [all_input_ids, all_segment_ids, all_input_mask]

            def fit(self, X, y=None, **fit_params):
                return self

        class MyBERT(torch.nn.Module):
            """Class to wrap the current BERT model."""
            def __init__(self):
                super(MyBERT, self).__init__()
                self.model = model

            def forward(self, X):
                """Apply a softmax function to the output of the BERT model."""
                return torch.nn.functional.softmax(self.model(*X), dim=1)

        # Creates a NeuralNetClassifier.
        if device == torch.device('cuda'):
            net = NeuralNetClassifier(MyBERT, device='cuda', max_epochs=0, lr=0.0, train_split=None)
        else:
            net = NeuralNetClassifier(MyBERT, max_epochs=0, lr=0.0, train_split=None)

        # Set up the pipeline.
        c = make_pipeline(BertConverter(), net)
        # To initialize the pipeline (does not train, because epochs=0).
        c.fit(raw_text_list, y=torch.zeros(len(raw_text_list), dtype=torch.long))

        # Print the predictions and probabilities for the example texts.
        print(c.predict_proba(raw_text_list))

        # Creates the LimeTextExplainer.
        # bow=True to replace all occurrences of a string at once.
        explainer = LimeTextExplainer(class_names=processor.get_labels(), bow=False, mask_string="[UNK]")

        # Explain the first example in the list and save the result using LIME.
        idx = 0
        exp = explainer.explain_instance(raw_text_list[idx], c.predict_proba)
        print('Document id: %d' % idx)
        print('Probability(support) =', c.predict_proba([raw_text_list[idx]])[0, 1])
        print('True class: %s' % "None")
        print(exp.as_list())
        exp.save_to_file(os.path.join(args.output_dir, "lime.html"))

        # Explain the first example using the ANCHOR explainer and save the result.
        nlp = spacy.load("en_core_web_sm")
        explainer2 = anchor_text.AnchorText(nlp, processor.get_labels(), use_unk_distribution=True)
        exp2 = explainer2.explain_instance(raw_text_list[idx], c.predict, threshold=0.95, use_proba=True)
        pred = explainer2.class_names[c.predict([raw_text_list[idx]])[0]]
        alternative = explainer2.class_names[1 - c.predict([raw_text_list[idx]])[0]]
        print('Anchor: %s' % (' AND '.join(exp2.names())))
        print('Precision: %.2f\n' % exp2.precision())
        print('Examples where anchor applies and model predicts %s:\n' % pred)
        print('\n'.join([x[0] for x in exp2.examples(only_same_prediction=True)]))
        print('Examples where anchor applies and model predicts %s:\n' % alternative)
        print('\n'.join([x[0] for x in exp2.examples(only_different_prediction=True)]))
        exp2.save_to_file(os.path.join(args.output_dir, "anchor.html"))


if __name__ == "__main__":
    """Command line program to fine-tune BERT."""
    main()
