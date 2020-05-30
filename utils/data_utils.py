#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:
  * @file: data_utils.py
  * @time: 2020/05/29
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import pickle
import collections

import tensorflow as tf


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, words=None, labels=None):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class NerProcessor(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, data_dir):
        return self._read_data(os.path.join(data_dir, "train.txt"), "train")

    def get_dev_examples(self, data_dir):
        return self._read_data(os.path.join(data_dir, "dev.txt"), "dev")

    def get_test_examples(self, data_dir):
        return self._read_data(os.path.join(data_dir, "test.txt"), "test")

    def get_labels(self):
        if os.path.exists(os.path.join(self.data_dir, "labels.txt")):
            labels = [_.strip() for _ in open(os.path.join(self.data_dir, "labels.txt"), "r").readlines() if _.strip()]
        else:
            labels = ["O", "B-DEF", "I-DEF", "B-PLA", "I-PLA", "B-OTH", "I-OTH", "[CLS]", "[SEP]"]

        return labels

    @staticmethod
    def _read_data(input_file, mode):
        """Reads a BIO data."""
        examples = []
        guid_index = 1
        with codecs.open(input_file, "r", encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
        return examples


def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, "a", encoding="utf-8")
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + "\n")
        wf.close()


def convert_single_example(
    ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode
):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, "label2id.pkl")):
        with codecs.open(os.path.join(output_dir, "label2id.pkl"), "wb") as w:
            pickle.dump(label_map, w)

    words = example.words
    label_list = example.labels
    tokens = []
    labels = []
    for i, word in enumerate(words):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        tmp_label = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(tmp_label)
            else:  # 一般不会出现else
                labels.append("X")

    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0: (max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0: (max_seq_length - 2)]
    base_tokens = []
    segment_ids = []
    label_ids = []
    base_tokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用CLS 也没毛病
    for i, token in enumerate(tokens):
        base_tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    base_tokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(base_tokens)  # 将序列中的字(base_tokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(label_map["O"])
        base_tokens.append("**NULL**")
        # label_mask.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join(words))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # mode='test'的时候才有效
    write_tokens(base_tokens, output_dir, mode)
    return feature


def filed_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None
):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(
            ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_calls=4,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                drop_remainder=drop_remainder,
            )
        )
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn
