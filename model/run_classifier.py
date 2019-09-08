# encoding: utf-8

import sys

sys.path.append('../')

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from bert import modeling
from bert import optimization
from bert import tokenization
from bert.tf_metrics import *
from bert.my_data_loader import MentionProcessor
from model.args import args_parse

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags

FLAGS = flags.FLAGS

# %%
## Required parameters
flags.DEFINE_string(
    "data_dir",
    None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.",
)

flags.DEFINE_string(
    "bert_config_file",
    None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.",
)

flags.DEFINE_string("task_name", "MRPC", "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None, "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory where the model checkpoints will be written.",
)

## Other parameters

flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Initial checkpoint (usually from a pre-trained BERT model).",
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.",
)

flags.DEFINE_integer(
    "max_seq_length",
    128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False, "Whether to run the model in inference mode on the test set."
)

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 64, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 2.0, "Total number of training epochs to perform."
)

flags.DEFINE_float(
    "warmup_proportion",
    0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.",
)

flags.DEFINE_integer(
    "save_checkpoints_steps", 50, "How often to save the model checkpoint."
)

flags.DEFINE_integer(
    "iterations_per_loop", 50, "How many steps to make in each estimator call."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name",
    None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.",
)

tf.flags.DEFINE_string(
    "tpu_zone",
    None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.",
)

tf.flags.DEFINE_string(
    "gcp_project",
    None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.",
)

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores",
    8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.",
)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
    create bert + dense model
    :param bert_config:
    :param is_training:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :param use_one_hot_embeddings:
    :return:
    """
    model = modeling.BertModel(config=bert_config, is_training=is_training, input_ids=input_ids,
                               input_mask=input_mask, token_type_ids=segment_ids, use_one_hot_embeddings=use_one_hot_embeddings)

    bert_output = model.get_pooled_output()
    hidden_size = bert_output.shape[-1]


    output_weight = tf.get_variable(name='output_weight', shape=[hidden_size, num_labels], initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_biase = tf.get_variable(name='output_biase', shape=[num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope('loss'):

        if is_training:
            bert_output = tf.nn.dropout(bert_output, keep_prob=0.9)

        logits = tf.matmul(bert_output, output_weight)
        logits = tf.nn.bias_add(logits, output_biase)
        possibility = tf.nn.softmax(logits, axis=-1)
        # compute log for cross entropy
        log_softmax = tf.nn.log_softmax(logits, axis=-1)

        one_hot_label = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        logits_loss = -tf.reduce_mean(log_softmax * one_hot_label, axis=-1)
        loss = tf.reduce_mean(logits_loss)

        return (loss, logits_loss, logits, possibility)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, logits_loss, logits, possibility = create_model(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, False)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                 modeling.get_assignment_map_from_checkpoint(tvars,
                                                             init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            #train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            train_op = optimization.create_optimizer(
                 total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            # hook_dict = {}
            # hook_dict['loss'] = total_loss
            # hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            # logging_hook = tf.train.LoggingTensorHook(
            #     hook_dict, every_n_iter=args.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                p = precision(label_ids, predictions, num_classes=3, average='macro')
                r = recall(label_ids, predictions, num_classes=3, average='macro')
                F1 = f1(label_ids, predictions, num_classes=3, average='macro')
                return {"precession": p, 'recall': r, 'f1': F1}


            eval_metrics = metric_fn(label_ids, logits)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)

        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=possibility
            )
        return output_spec

    return model_fn


# def _decode_record(record, name_to_features):
#     """Decodes a record to a TensorFlow example."""
#     example = tf.parse_single_example(record, name_to_features)
#
#     # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
#     # So cast all int64 to int32.
#     for name in list(example.keys()):
#         t = example[name]
#         if t.dtype == tf.int64:
#             t = tf.to_int32(t)
#         example[name] = t
#
#     return example
#
#
# def read_data(data, batch_size, is_training, max_seq_length, num_epochs=1):
#
#     name_to_features = {
#         "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
#         "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
#         "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
#         "label_ids": tf.FixedLenFeature([], tf.int64),
#     }
#
#     # For training, we want a lot of parallel reading and shuffling.
#     # For eval, we want no shuffling and parallel reading doesn't matter.
#
#     if is_training:
#         data = data.shuffle(buffer_size=10000)
#         data = data.repeat(num_epochs)
#
#     data = data.apply(
#         tf.contrib.data.map_and_batch(
#             lambda record: _decode_record(record, name_to_features),
#             batch_size=batch_size))
#
#     return data


def filed_based_convert_examples_to_features(input_file, max_seq_length, is_training, batch_size, num_epochs=1):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example


    def input_fn():
        dataset = tf.data.TFRecordDataset(args.rd_train_path)

        if is_training:
            dataset = dataset.shuffle(buffer_size=100)
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size))

        return dataset

    return input_fn


def train(args, epoch_num=1):
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = os.path.join(args.output_path, time_stamp)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    bert_config = modeling.BertConfig.from_json_file(args.bert_config_path)
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=output_path,
        save_summary_steps=200,
        save_checkpoints_steps=200,
        session_config=session_config,
        keep_checkpoint_max=2
    )

    if args.max_seq_length > bert_config.max_position_embeddings:  # the max position index is 512
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    processor = MentionProcessor()
    labels = [0, 1, 2]
    train_examples = processor.get_examples(args.train_path, args.label_path)

    if args.do_train and args.do_eval:
        train_example_num = len(train_examples)
        train_steps = train_example_num // args.batch_size * epoch_num
        num_warmup_step = int(train_steps * args.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(labels),
        init_checkpoint=args.checkpoint_path,
        learning_rate=args.learning_rate,
        num_train_steps=train_steps,
        num_warmup_steps=num_warmup_step
    )
# the train sample num is 6538
    estimator = tf.estimator.Estimator(model_fn, config=run_config)

    if args.do_train and args.do_eval:
        # train_record_data = tf.data.TFRecordDataset(args.rd_train_path)
        # train_input_fn = read_data(train_record_data, args.batch_size, is_training=True, max_seq_length=512, num_epochs=5)
        train_input_fn = filed_based_convert_examples_to_features(args.rd_train_path, args.max_seq_length,
                                                                  True, args.batch_size, epoch_num)

        # eval_record_data = tf.data.TFRecordDataset(args.rd_dev_path)
        # dev_input_fn = read_data(eval_record_data, args.batch_size, is_training=False, max_seq_length=512)
        dev_input_fn = filed_based_convert_examples_to_features(args.rd_dev_path, args.max_seq_length, True, args.batch_size, 1)

        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=1000,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=500)


        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps, hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, start_delay_secs=600, throttle_secs=600)
        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def test_dataset(args):
    train_input_fn = filed_based_convert_examples_to_features(args.rd_train_path, args.max_seq_length,
                                                              True, args.batch_size, 1)

    train_datatset = train_input_fn()
    itetor = train_datatset.make_one_shot_iterator()
    train_batch =itetor.get_next()
    with tf.Session() as sess:
        while True:
            dataset = sess.run(train_batch)
            print(dataset)


if __name__ == '__main__':
    args = args_parse()
    #
    train(args, epoch_num=10)
    # test_dataset(args)







