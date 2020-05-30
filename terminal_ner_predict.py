#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:zbl
  * @file: terminal_ner_predict.py
  * @time: 2020/05/30
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import pickle
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser

from model.get_model import create_model
from utils.data_utils import InputFeatures

args = get_args_parser()

model_dir = "./output-ner-law"
bert_dir = "/run/media/zbl/works/python/model-data/bert_hinese_L-12_H-768_A-12/tf1.0"

is_training = False
use_one_hot_embeddings = False
batch_size = 1
args.max_seq_length = 512

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
model = None

# input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None

print("checkpoint path:{}".format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, "label2id.pkl"), "rb") as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

num_labels = len(id2label.keys()) + 1
# exit()

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, "bert_config.json"))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids_p,
        input_mask=input_mask_p,
        segment_ids=None,
        labels=None,
        num_labels=num_labels,
        use_one_hot_embeddings=False,
        dropout_rate=1.0,
    )

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, "vocab.txt"), do_lower_case=args.do_lower_case)

ckpt = tf.train.get_checkpoint_state(model_dir)
ckpt_path = ckpt.model_checkpoint_path


def read_model_param_and_value():
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    param_dict = reader.get_variable_to_shape_map()

    for key, val in param_dict.items():
        try:
            # if "crf_loss" in key or "project" in key:
            # print(key)  # , reader.get_tensor(key))
            if "bert/encoder/Reshape_13" in key:
                print(key, reader.get_tensor(key))
        except:
            pass
    exit()


def predict_online():

    def convert(line):
        feature = convert_single_example(0, line, args.max_seq_length, tokenizer, "p")
        input_ids = np.reshape([feature.input_ids], (1, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (1, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (1, args.max_seq_length))
        label_ids = np.reshape([feature.label_ids], (1, args.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    # 【连续调整盼援军 增量资金在何方】近期市场持续调整，昨日两市更是放量下跌，上证指数跌幅达到2.43%，深证指数跌幅达到3.21%，创业板指跌幅达到2.84%。下跌背后，市场进入关键时点，在强压力区连续调整，市场的上涨仍需增量资金的进入。（中国证券报）
    # global graph
    with graph.as_default():
        print(id2label)
        while True:
            print("input the test sentence:")
            sentence = str(input())
            if sentence == "q":
                break
            if len(sentence) < 2:
                continue
            sentence = tokenizer.tokenize(sentence)
            print("your input is:{}".format(sentence))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)
            print(
                ", ".join([str(_) for _ in input_ids[0]]),
                "\n",
                ", ".join([str(_) for _ in input_mask[0]]),
                "\n",
                ", ".join([str(_) for _ in label_ids[0]]),
            )
            _input_ids = np.array([input_ids] * batch_size)
            _input_ids = np.reshape(_input_ids, [batch_size, args.max_seq_length])

            _input_mask = np.array([input_mask] * batch_size)
            _input_mask = np.reshape(_input_mask, [batch_size, args.max_seq_length])

            _label_ids = np.array([label_ids] * batch_size)
            _label_ids = np.reshape(_label_ids, [batch_size, args.max_seq_length])

            feed_dict = {input_ids_p: _input_ids, input_mask_p: _input_mask}
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            print(pred_label_result, pred_ids)
            strage_combined_link_org_loc(sentence, pred_label_result[0])


def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ["[CLS]", "[SEP]"]:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """

    def print_output(data, type):
        line = [type]
        for i in data:
            # print(i.types, i.start, i.end)
            line.append(i.word)
        print(", ".join(line))

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[: len(tags)]
    defs, pla, oth = eval.get_result(tokens, tags)
    print_output(defs, "DEF")
    print_output(pla, "PLA")
    print_output(oth, "OTH")


def convert_single_example(
    ex_index, example, max_seq_length, tokenizer, mode
):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = label2id
    # # 1表示从1开始对label进行index化
    # for (i, label) in enumerate(label_list, 1):
    #     label_map[label] = i
    # # 保存label->index 的map
    # if not os.path.exists(os.path.join(model_dir, "label2id.pkl")):
    #     with codecs.open(os.path.join(model_dir, "label2id.pkl"), "wb") as w:
    #         pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0 : (max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def merge(self):
        return self.__merge

    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types

    @word.setter
    def word(self, word):
        self.__word = word

    @start.setter
    def start(self, start):
        self.__start = start

    @end.setter
    def end(self, end):
        self.__end = end

    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append("entity:{}".format(self.__word))
        line.append("start:{}".format(self.__start))
        line.append("end:{}".format(self.__end))
        line.append("merge:{}".format(self.__merge))
        line.append("types:{}".format(self.__types))
        return "\t".join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.defs = []
        self.pla = []
        self.oth = []
        self.others = []

    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.defs, self.pla, self.oth

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ""

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx + 1, tag[2:])
                item["entities"].append(
                    {"word": char, "start": idx, "end": idx + 1, "type": tag[2:]}
                )
            elif tag[0] == "B":
                if entity_name != "":
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append(
                        {
                            "word": entity_name,
                            "start": entity_start,
                            "end": idx,
                            "type": last_tag[2:],
                        }
                    )
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != "":
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append(
                        {
                            "word": entity_name,
                            "start": entity_start,
                            "end": idx,
                            "type": last_tag[2:],
                        }
                    )
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != "":
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append(
                {
                    "word": entity_name,
                    "start": entity_start,
                    "end": idx,
                    "type": last_tag[2:],
                }
            )
        return item

    def append(self, word, start, end, tag):
        if tag == "DEF":
            self.defs.append(Pair(word, start, end, "DEF"))
        elif tag == "PLA":
            self.pla.append(Pair(word, start, end, "PLA"))
        elif tag == "OTH":
            self.oth.append(Pair(word, start, end, "OTH"))
        else:
            self.others.append(Pair(word, start, end, tag))


if __name__ == "__main__":
    predict_online()
