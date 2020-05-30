#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:
  * @file: get_model.py
  * @time: 2020/05/29
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from bert_base.bert import modeling, optimization
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers


class BiLstmCRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, labels, lengths, is_training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training

    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # blstm
            lstm_output = self.blstm_layer(self.embedded_chars)
            # project
            logits = self.project_bilstm_layer(lstm_output)
        # crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return loss, logits, trans, pred_ids

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars, dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels,
                 use_one_hot_embeddings, dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # 添加CRF output layer(其实可以不加，最后的效果也差不多，而且在做模型压缩转化的时候会遇到很多问题)
    blstm_crf = BiLstmCRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
    return rst


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, args):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print("shape of input_ids", input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False
            )
            hook_dict = {"loss": total_loss, "global_steps": tf.train.get_or_create_global_step()}
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.save_summary_steps
            )

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(
                        labels=label_ids, predictions=pred_ids
                    )
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_ids)
        return output_spec

    return model_fn
