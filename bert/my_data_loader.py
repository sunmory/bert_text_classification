# encoding: utf-8

import os
import csv
import collections
import pandas as pd
import tensorflow as tf
from bert import tokenization


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

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MentionProcessor(DataProcessor):
    def __init__(self):
        pass

    def get_examples(self, data_path, label_path):
        """
        read sample for original dataset and store in class InputExample
        :param data_path:
        :param label_path:
        :return:
        """
        examples = []
        dataframe = pd.read_csv(data_path, encoding='utf-8', index_col='id')
        labelframe = pd.read_csv(label_path, encoding='utf-8', index_col='id')
        for i in range(dataframe.shape[0]):
            text_id = dataframe.index[i]
            text = dataframe.loc[text_id, 'title'] + dataframe.loc[text_id, 'content']
            # text = text[:max_seq_length]
            guid = "train-%d" % (i)
            text = tokenization.convert_to_unicode(text)
            label = labelframe.loc[text_id, 'label']
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label)
            )
        return examples

    def convert_single_example(self, ex_index, example, label_list, max_seq_length, tokenizer):

        token_a = tokenizer.tokenize(example.text_a)
        token_b = None
        if example.text_b is not None:
            token_b = tokenizer.tokenize(example.text_a)
        if token_b:
            self._truncate_seq_pair(token_a, token_b, max_seq_length - 3)
        else:
            if len(token_a) > max_seq_length - 2:
                token_a = token_a[:max_seq_length - 2]

        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in token_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        if token_b:
            for token in token_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_masks.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_masks) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_masks]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        feature = InputFeatures(input_ids=input_ids, input_mask=input_masks, segment_ids=segment_ids, label_id=int(label_id))
        return feature

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

    def file_based_convert_examples_to_features(self,  examples, label_list, max_seq_length, tokenizer, output_file):
        writer = tf.python_io.TFRecordWriter(output_file)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

            rd_feature = collections.OrderedDict()
            rd_feature['input_ids'] = create_int_feature(feature.input_ids)
            rd_feature['input_mask'] = create_int_feature(feature.input_mask)
            rd_feature['segment_ids'] = create_int_feature(feature.segment_ids)
            rd_feature['label_ids'] = create_int_feature([feature.label_id])


            rd_example = tf.train.Example(features=tf.train.Features(feature=rd_feature))

            writer.write(rd_example.SerializeToString())


