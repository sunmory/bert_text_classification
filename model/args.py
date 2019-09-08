# encoding: utf-8

import argparse


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../dataset/train.csv', help='train_path')
    parser.add_argument('--dev_path', type=str, default='../dataset/dev.csv', help='dev_path')
    parser.add_argument('--new_train_path', type=str, default='../dataset/new_train.csv', help='train_path')
    parser.add_argument('--new_label_path', type=str, default='../dataset/new_label.csv', help='dev_path')
    parser.add_argument('--label_path', type=str, default='../dataset/label.csv', help='label_path')
    parser.add_argument('--rd_train_path', type=str, default='../dataset/train.tf_record', help='rd_train_path')
    parser.add_argument('--rd_dev_path', type=str, default='../dataset/dev.tf_record', help='rd_dev_path')
    parser.add_argument('--new_rd_train', type=str, default='../dataset/new_train.tf_record', help='rd_train_path')
    # parser.add_argument('--new_rd_dev', type=str, default='../dataset/new_dev.tf_record', help='rd_dev_path')
    parser.add_argument('--vocab_file_path', type=str, default='../chinese_L-12_H-768_A-12/vocab.txt', help='vocab_file_path')

    parser.add_argument('--max_seq_length', type=int, default=512, help='max_seq_length')
    parser.add_argument('--bert_config_path', type=str, default='../chinese_wwm_ext_L-12_H-768_A-12/bert_config.json', help='bert_config_path')
    parser.add_argument('--checkpoint_path', type=str, default='../chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt', help='checkpoint_path')
    parser.add_argument('--batch_size', type=int, default=5, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--output_path', type=str, default='../output', help='output_path')
    parser.add_argument('--do_train', type=bool, default=True, help='do_train')
    parser.add_argument('--do_eval', type=bool, default=True, help='do_eval')
    parser.add_argument('--do_predict', type=bool, default=False, help='do_predict')

    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup_proportion')
    args = parser.parse_args()
    return args