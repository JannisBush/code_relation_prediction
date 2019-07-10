# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2019 Jannis Rautenstrauch
#
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
""" BERT classification fine-tuning: utilities to work with Relation Prediction tasks """

from __future__ import absolute_import, division, print_function

import logging
import os

import pandas as pd

from functools import partial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, input_to_use):
        """Saves the input to use."""
        self.input_to_use = input_to_use

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_splits(self, data_dir, splits):
        """Gets a collection of `InputExample`s for the complete data set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def create_inputs(self, dataset):
        """Creates the Input Examples for a dataset."""
        if self.input_to_use == "both":
            return dataset.apply(
                lambda x: InputExample(guid=None, text_a=x["org"], text_b=x["response"], label=x["label"]),
                axis=1), dataset
        elif self.input_to_use == "org":
            return dataset.apply(
                lambda x: InputExample(guid=None, text_a=x["org"], text_b=None, label=x["label"]),
                axis=1), dataset
        elif self.input_to_use == "response":
            return dataset.apply(
                lambda x: InputExample(guid=None, text_a=x["response"], text_b=None, label=x["label"]),
                axis=1), dataset
        else:
            raise ValueError("Invalid input_to_use, has to be one of both, org or response.")


class NoDEProcessor(DataProcessor):
    """Processor for the NoDE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "test")

    def get_splits(self, data_dir, splits=2):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, "complete_data.tsv"), sep='\t')
        dataset = df.loc[df['org_dataset'].isin(['debate_train', 'debate_test', 'procon'])]
        skf = StratifiedKFold(n_splits=splits, random_state=113)
        splits_data = []
        for train_idx, val_idx in skf.split(dataset, dataset['label']):
            train_examples, train_df = self.create_inputs(dataset.iloc[train_idx])
            test_examples, test_df = self.create_inputs(dataset.iloc[val_idx])
            splits_data.append((train_examples, train_df,
                                test_examples, test_df))
        return splits_data

    def get_labels(self):
        """See base class."""
        return ["attack", "support"]

    def _create_examples(self, filename, set_type):
        """Creates examples for the training and test sets."""
        df = pd.read_csv(filename, sep='\t')
        if set_type == "train":
            dataset = df.loc[df['org_dataset'].isin(['debate_train', 'procon'])]
        elif set_type == "test":
            dataset = df.loc[df['org_dataset'].isin(['debate_test'])]
        else:
            raise ValueError("Invalid set_type, has to be one of train, test or both.")
        return self.create_inputs(dataset)


class PoliticalProcessor(DataProcessor):
    """Processor for the Political data set."""

    def __init__(self, task, input_to_use):
        """Sets the task and mode for the Political dataset."""
        super().__init__(input_to_use)
        self.task = task

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "test")

    def get_splits(self, data_dir, splits=10):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, "complete_data.tsv"), sep='\t')
        df = df.loc[df['org_dataset'].isin(['political'])]
        if self.task == "AS":
            df = df[df['label'] != 'unrelated']
        elif self.task == "RU":
            df = df.replace({'label': {'attack': 'related', 'support': 'related'}})
        dataset = df
        skf = StratifiedKFold(n_splits=splits, random_state=113)
        splits_data = []
        for train_idx, val_idx in skf.split(dataset, dataset['label']):
            train_examples, train_df = self.create_inputs(dataset.iloc[train_idx])
            test_examples, test_df = self.create_inputs(dataset.iloc[val_idx])
            splits_data.append((train_examples, train_df,
                                test_examples, test_df))
        return splits_data

    def get_labels(self):
        """See base class."""
        if self.task == "AS":
            return ["attack", "support"]
        elif self.task == "RU":
            return ["related", "unrelated"]
        elif self.task == "ASU":
            return ["attack", "support", "unrelated"]
        else:
            raise ValueError("Invalid task, has to be one of AS, RU or ASU.")

    def _create_examples(self, filename, set_type):
        """Creates examples for the training and test sets."""
        df = pd.read_csv(filename, sep='\t')
        df = df.loc[df['org_dataset'].isin(['political'])]
        if self.task == "AS":
            df = df[df['label'] != 'unrelated']
        elif self.task == "RU":
            df = df.replace({'label': {'attack': 'related', 'support': 'related'}})
        train, test = train_test_split(df, test_size=0.2, random_state=113, stratify=df['label'])
        if set_type == "train":
            dataset = train
        elif set_type == "test":
            dataset = test
        else:
            raise ValueError("Invalid set_type, has to be one of train, test or both.")
        return self.create_inputs(dataset)


class AgreementProcessor(DataProcessor):
    """Processor for the Agreement/Disagreement Dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "test")

    def get_splits(self, data_dir, splits=10):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, "complete_data.tsv"), sep='\t')
        dataset = df.loc[df['org_dataset'].isin(['agreement'])]
        skf = StratifiedKFold(n_splits=splits, random_state=113)
        splits_data = []
        for train_idx, val_idx in skf.split(dataset, dataset['label']):
            train_examples, train_df = self.create_inputs(dataset.iloc[train_idx])
            test_examples, test_df = self.create_inputs(dataset.iloc[val_idx])
            splits_data.append((train_examples, train_df,
                                test_examples, test_df))
        return splits_data

    def get_labels(self):
        """See base class."""
        return ["agreement", "disagreement"]

    def _create_examples(self, filename, set_type):
        """Creates examples for the training and test sets."""
        df = pd.read_csv(filename, sep='\t')
        df = df.loc[df['org_dataset'].isin(['agreement'])]
        train, test = train_test_split(df, test_size=0.2, random_state=113, stratify=df['label'])
        if set_type == "train":
            dataset = train
        elif set_type == "test":
            dataset = test
        else:
            raise ValueError("Invalid set_type, has to be one of train, test or both.")
        return self.create_inputs(dataset)


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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


def simple_accuracy(preds, labels):
    """Returns the accuracy of the prediction."""
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    """Returns the accuracy and the f1 score of the prediction."""
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    class_rep = classification_report(y_true=labels, y_pred=preds, output_dict=True)
    conf_mat = confusion_matrix(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "classification_report": class_rep,
        "confusion_matrix": conf_mat
    }


def compute_metrics(task_name, preds, labels):
    """Computes and returns the correct metric for the given task."""
    assert len(preds) == len(labels)
    if task_name == "node":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name in ["political-as", "political-ru", "political-asu", "agreement"]:
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    else:
        raise KeyError(task_name)


processors = {
    "node": NoDEProcessor,
    "political-as": partial(PoliticalProcessor, "AS"),
    "political-ru": partial(PoliticalProcessor, "RU"),
    "political-asu": partial(PoliticalProcessor, "ASU"),
    "agreement": AgreementProcessor,
}
