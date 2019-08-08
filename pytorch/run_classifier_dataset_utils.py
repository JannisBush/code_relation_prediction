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
import random

import pandas as pd

from functools import partial
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import f1_score

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
        """Creates the DataProcessor and saves what to use as input."""
        self.input_to_use = input_to_use

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_splits(self, data_dir, splits):
        """Gets a collection of `InputExample`s for the complete data set divided in splits train and dev sets."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def create_inputs(self, dataset):
        """Creates the Input Examples for a dataset depending on the input_to_use variable."""
        # Normal mode: use the original argument as A and the response argument as B.
        if self.input_to_use == "both":
            return dataset.apply(
                lambda x: InputExample(guid=None, text_a=x["org"], text_b=x["response"], label=x["label"]),
                axis=1), dataset
        # Only org mode: use the original argument as A.
        elif self.input_to_use == "org":
            return dataset.apply(
                lambda x: InputExample(guid=None, text_a=x["org"], text_b=None, label=x["label"]),
                axis=1), dataset
        # Only response mode: use the response argument as A.
        elif self.input_to_use == "response":
            return dataset.apply(
                lambda x: InputExample(guid=None, text_a=x["response"], text_b=None, label=x["label"]),
                axis=1), dataset
        # Reversed mode: use the response argument as A and the original argument as B.
        elif self.input_to_use == "response-org":
            return dataset.apply(
                lambda x: InputExample(guid=None, text_a=x["response"], text_b=x["org"], label=x["label"]),
                axis=1), dataset
        # Invalid mode, throw an error.
        else:
            raise ValueError("Invalid input_to_use, has to be one of both, org or response.")

    def _create_normal_splits(self, dataset, splits):
        """Creates the default StratifiedKFold splits for dataset."""
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=113)
        splits_data = []
        for train_idx, val_idx in skf.split(dataset, dataset['label']):
            train_examples, train_df = self.create_inputs(dataset.iloc[train_idx])
            test_examples, test_df = self.create_inputs(dataset.iloc[val_idx])
            splits_data.append((train_examples, train_df,
                                test_examples, test_df))
        return splits_data


class NoDEProcessor(DataProcessor):
    """Processor for the NoDE data set."""

    def __init__(self, train_data_names, input_to_use):
        """Sets the train_data and mode for the NoDE dataset."""
        super().__init__(input_to_use)
        self.train_data_names = train_data_names

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "dev")

    def get_splits(self, data_dir, splits=2):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, "complete_data.tsv"), sep='\t')
        dataset = df.loc[df['org_dataset'].isin(self.train_data_names + ['debate_test'])]
        return self._create_normal_splits(dataset, splits)

    def get_labels(self):
        """See base class."""
        return ["attack", "support"]

    def _create_examples(self, filename, set_type):
        """Creates examples for the training and test sets."""
        df = pd.read_csv(filename, sep='\t')
        if set_type == "train":
            dataset = df.loc[df['org_dataset'].isin(self.train_data_names)]
        elif set_type == "dev":
            dataset = df.loc[df['org_dataset'].isin(['debate_test'])]
        else:
            raise ValueError("Invalid set_type, has to be one of train or dev.")
        return self.create_inputs(dataset)


class PoliticalProcessor(DataProcessor):
    """Processor for the Political data set."""

    def __init__(self, task, cv_method, input_to_use):
        """Sets the task and mode and cv-method for the Political dataset."""
        super().__init__(input_to_use)
        self.task = task
        # Only gets used in get_splits
        self.cv_method = cv_method

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "dev")

    def get_splits(self, data_dir, splits=10):
        """See base class."""
        df = self._get_df(os.path.join(data_dir, "complete_data.tsv"))
        # The original CV method, stratifiedKFold.
        if self.cv_method == 'original':
            splits_data = self._create_normal_splits(df, splits)
        # Leave One Group Out CV, Groups=Topics.
        elif self.cv_method == 'topics':
            logo = LeaveOneGroupOut()
            splits_data = []
            for train_idx, val_idx in logo.split(df, groups=df['topic']):
                train_examples, train_df = self.create_inputs(df.iloc[train_idx])
                test_examples, test_df = self.create_inputs(df.iloc[val_idx])
                splits_data.append((train_examples, train_df,
                                    test_examples, test_df))
        # Other wise invalid cv_method, throw an error.
        else:
            raise ValueError("Invalid cv_method, has to be original or topics")
        # Return the splits
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
        df = self._get_df(filename)
        # Train test split with 20% test size.
        train, test = train_test_split(df, test_size=0.2, random_state=113, stratify=df['label'])
        if set_type == "train":
            dataset = train
        elif set_type == "dev":
            dataset = test
        else:
            raise ValueError("Invalid set_type, has to be one of train or test.")
        return self.create_inputs(dataset)

    def _get_df(self, filename):
        """Returns the dataframe according to the task."""
        df = pd.read_csv(filename, sep='\t')
        df = df.loc[df['org_dataset'].isin(['political'])]
        # Ignore the unrelated pairs in AS mode.
        if self.task == "AS":
            df = df[df['label'] != 'unrelated']
        # Merge attack and support pairs in RU mode.
        elif self.task == "RU":
            df = df.replace({'label': {'attack': 'related', 'support': 'related'}})
        return df


class AgreementProcessor(DataProcessor):
    """Processor for the Agreement/Disagreement Dataset."""

    def __init__(self, split_topics_remove_duplicates, input_to_use):
        """Sets the mode for the Agreement dataset."""
        super().__init__(input_to_use)
        # Only gets uses in get_train_examples and get_dev_examples, ignored in CV.
        self.split_topics_remove_duplicates = split_topics_remove_duplicates

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "complete_data.tsv"), "dev")

    def get_splits(self, data_dir, splits=10):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, "complete_data.tsv"), sep='\t')
        dataset = df.loc[df['org_dataset'].isin(['agreement'])]
        return self._create_normal_splits(dataset, splits)

    def get_labels(self):
        """See base class."""
        return ["agreement", "disagreement"]

    def _create_examples(self, filename, set_type):
        """Creates examples for the training and test sets."""
        df = pd.read_csv(filename, sep='\t')
        df = df.loc[df['org_dataset'].isin(['agreement'])]
        # Random 80/20 train/test split.
        if not self.split_topics_remove_duplicates:
            train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=113, stratify=df['label'])
        # Fair mode: 80% of topics in train set, 20% of topics in test set.
        else:
            # Removes all text duplicates.
            df = df.drop_duplicates(subset=['org', 'response'])
            # Sort all topics and deterministic shuffle.
            topics = sorted(set(df['topic'].values))
            random.Random(1).shuffle(topics)
            # First 80% of topics are the train set.
            train = df.loc[df['topic'].isin(topics[:130])].sample(frac=1, random_state=113)
            # Last 20% of topics are the test set.
            test = df.loc[df['topic'].isin(topics[130:])].sample(frac=1, random_state=113)
        if set_type == "train":
            dataset = train
        elif set_type == "dev":
            dataset = test
        else:
            raise ValueError("Invalid set_type, has to be one of train or dev.")
        return self.create_inputs(dataset)


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a list of `InputExample`s into a list of `InputFeatures`s."""

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

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    # Convert all examples into features.
    for (ex_index, example) in enumerate(examples):
        # Log info about progress all 10000 examples
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # Tokenize text A.
        tokens_a = tokenizer.tokenize(example.text_a)
        # Tokenize text B (if existent).
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

        # Add the [CLS] and [SEP] token and the segment A(0) tokens to text A.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        # Add the [SEP] token and the segment B(1) tokens to text B.
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        # Convert the tokens to ids.
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create a mask to only attend to real tokens.
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the maximum sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # Check if all padded inputs have length equal to the maximum sequence length.
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # Convert the label to the corresponding label id.
        label_id = label_map[example.label]

        # Log information about the first 5 examples.
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        # Create and append the input feature.
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    # Return the list of all input features.
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
    """Returns the accuracy and the weighted f1 score of the prediction."""
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return {
        "acc": acc,
        "f1": f1,
    }


def compute_metrics(task_name, preds, labels):
    """Computes and returns the correct metric for the given task."""
    assert len(preds) == len(labels)
    if task_name in ["node", "node-ext"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name in ["political-as", "political-ru", "political-asu",
                       "political-as-topics", "political-ru-topics", "political-asu-topics",
                       "agreement", "agreement-topics"]:
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "node": partial(NoDEProcessor, ['debate_train']),
    "node-ext": partial(NoDEProcessor, ['debate_train', 'procon']),
    "political-as": partial(PoliticalProcessor, "AS", "original"),
    "political-ru": partial(PoliticalProcessor, "RU", "original"),
    "political-asu": partial(PoliticalProcessor, "ASU", "original"),
    "political-as-topics": partial(PoliticalProcessor, "AS", "topics"),
    "political-ru-topics": partial(PoliticalProcessor, "RU", "topics"),
    "political-asu-topics": partial(PoliticalProcessor, "ASU", "topics"),
    "agreement": partial(AgreementProcessor, False),
    "agreement-topics": partial(AgreementProcessor, True),
}
