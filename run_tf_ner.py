#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:zbl
  * @file: run_tf_ner.py
  * @time: 2020/05/29
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import csv
import codecs
import pickle
import time
import os
import tensorflow as tf
from bert_base.bert import tokenization, optimization, modeling
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers
import logging

logging.getLogger().setLevel(tf.logging.INFO)

flags = tf.flags

FLAGS = flags.FLAGS

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

flags.DEFINE_string("task_name", default="ner", help="The name of the task to train.")

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

flags.DEFINE_integer(
    "max_seq_length",
    128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True, "Whether to run the model in inference mode on the test set."
)

flags.DEFINE_integer(
    "batch_size", default=64, help="Total batch size for training, eval and predict."
)
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 10.0, "Total number of training epochs to perform."
)
flags.DEFINE_float("dropout_rate", default=0.5, help="Dropout rate")
flags.DEFINE_float(
    "warmup_proportion",
    0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.",
)
flags.DEFINE_float("clip", default=0.5, help="Gradient clip")
flags.DEFINE_integer("lstm_size", default=128, help="size of lstm units.")
flags.DEFINE_integer(
    "num_layers", default=1, help="number of rnn layers, default is 1."
)
flags.DEFINE_string("cell", default="lstm", help="which rnn cell used.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000, "How often to save the model checkpoint."
)

flags.DEFINE_integer("save_summary_steps", 1000, "save_summary_steps")
flags.DEFINE_bool(
    "do_lower_case", default=True, help="Whether to lower case the input text."
)
flags.DEFINE_bool("clean", default=True, help="")
flags.DEFINE_string("device_map", default="0", help="witch device using to train")

# add labels
flags.DEFINE_string(
    "label_list",
    default=None,
    help="User define labels， can be a file with one label one line or a string using ',' split",
)


class BLSTM_CRF(object):
    def __init__(
        self,
        embedded_chars,
        hidden_unit,
        cell_type,
        num_layers,
        dropout_rate,
        initializers,
        num_labels,
        seq_length,
        labels,
        lengths,
        is_training,
    ):
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
        pred_ids, _ = crf.crf_decode(
            potentials=logits, transition_params=trans, sequence_length=self.lengths
        )
        return (loss, logits, trans, pred_ids)

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == "lstm":
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == "gru":
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
        with tf.variable_scope("rnn_layer"):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell(
                    [cell_fw] * self.num_layers, state_is_tuple=True
                )
                cell_bw = rnn.MultiRNNCell(
                    [cell_bw] * self.num_layers, state_is_tuple=True
                )

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, embedding_chars, dtype=tf.float32
            )
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
                W = tf.get_variable(
                    "W",
                    shape=[self.hidden_unit * 2, self.hidden_unit],
                    dtype=tf.float32,
                    initializer=self.initializers.xavier_initializer(),
                )

                b = tf.get_variable(
                    "b",
                    shape=[self.hidden_unit],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable(
                    "W",
                    shape=[self.hidden_unit, self.num_labels],
                    dtype=tf.float32,
                    initializer=self.initializers.xavier_initializer(),
                )

                b = tf.get_variable(
                    "b",
                    shape=[self.num_labels],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )

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
                W = tf.get_variable(
                    "W",
                    shape=[self.embedding_dims, self.num_labels],
                    dtype=tf.float32,
                    initializer=self.initializers.xavier_initializer(),
                )

                b = tf.get_variable(
                    "b",
                    shape=[self.num_labels],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                output = tf.reshape(
                    self.embedded_chars, shape=[-1, self.embedding_dims]
                )  # [batch_size, embedding_dims]
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
                initializer=self.initializers.xavier_initializer(),
            )
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths,
                )
                return tf.reduce_mean(-log_likelihood), trans


def create_model(
    bert_config,
    is_training,
    input_ids,
    input_mask,
    segment_ids,
    labels,
    num_labels,
    use_one_hot_embeddings,
    dropout_rate=1.0,
    lstm_size=1,
    cell="lstm",
    num_layers=1,
):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    import tensorflow as tf
    from bert_base.bert import modeling

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
    lengths = tf.reduce_sum(
        used, reduction_indices=1
    )  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # 添加CRF output layer
    blstm_crf = BLSTM_CRF(
        embedded_chars=embedding,
        hidden_unit=lstm_size,
        cell_type=cell,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        initializers=initializers,
        num_labels=num_labels,
        seq_length=max_seq_length,
        labels=labels,
        lengths=lengths,
        is_training=is_training,
    )
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
    return rst


def model_fn_builder(
    bert_config,
    num_labels,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    FLAGS,
):
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
            bert_config,
            is_training,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            num_labels,
            False,
            FLAGS.dropout_rate,
            FLAGS.lstm_size,
            FLAGS.cell,
            FLAGS.num_layers,
        )

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # 打印变量名
        # logger.info("**** Trainable Variables ****")
        #
        # # 打印加载模型的参数
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     logger.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False
            )
            hook_dict = {}
            hook_dict["loss"] = total_loss
            hook_dict["global_steps"] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=FLAGS.save_summary_steps
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




def main(_):
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device_map

    processors = {"ner": NerProcessor}
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d"
            % (FLAGS.max_seq_length, bert_config.max_position_embeddings)
        )

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):

            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                print(e)
                print("pleace remove the files of output dir and data.conf")
                exit(-1)

    # check output dir exists
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    processor = processors[FLAGS.task_name](FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case
    )

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True,
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=session_config,
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train and FLAGS.do_eval:
        # 加载训练数据
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) * 1.0 / FLAGS.batch_size * FLAGS.num_train_epochs
        )
        if num_train_steps < 1:
            raise AttributeError("training data is so small...")
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        # 打印验证集数据信息
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

    label_list = processor.get_labels(os.path.join(FLAGS.data_dir, "labels.txt"))
    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        FLAGS=FLAGS,
    )

    params = {"batch_size": FLAGS.batch_size}

    estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)

    if FLAGS.do_train and FLAGS.do_eval:
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(
                train_examples,
                label_list,
                FLAGS.max_seq_length,
                tokenizer,
                train_file,
                FLAGS.output_dir,
            )

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
        )
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(
                eval_examples,
                label_list,
                FLAGS.max_seq_length,
                tokenizer,
                eval_file,
                FLAGS.output_dir,
            )

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False,
        )

        # train and eval togither
        # early stop hook
        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name="loss",
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=FLAGS.save_checkpoints_steps,
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=num_train_steps,
            hooks=[early_stopping_hook],
        )
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(FLAGS.output_dir, "label2id.pkl"), "rb") as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(
            predict_examples,
            label_list,
            FLAGS.max_seq_length,
            tokenizer,
            predict_file,
            FLAGS.output_dir,
            mode="test",
        )

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
        )

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

        def result_to_pair(writer):
            for predict_line, prediction in zip(predict_examples, result):
                idx = 0
                line = ""
                line_token = str(predict_line.text).split(" ")
                label_token = str(predict_line.label).split(" ")
                len_seq = len(label_token)
                if len(line_token) != len(label_token):
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                    break
                for id in prediction:
                    if idx >= len_seq:
                        break
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ["[CLS]", "[SEP]"]:
                        continue
                    try:
                        line += (
                            line_token[idx]
                            + " "
                            + label_token[idx]
                            + " "
                            + curr_labels
                            + "\n"
                        )
                    except Exception as e:
                        tf.logging.info(e)
                        tf.logging.info(predict_line.text)
                        tf.logging.info(predict_line.label)
                        line = ""
                        break
                    idx += 1
                writer.write(line + "\n")

        with codecs.open(output_predict_file, "w", encoding="utf-8") as writer:
            result_to_pair(writer)
        from bert_base.train import conlleval

        eval_result = conlleval.return_report(output_predict_file)
        print("".join(eval_result))
        # 写结果到文件中
        with codecs.open(
            os.path.join(FLAGS.output_dir, "predict_score.txt"), "a", encoding="utf-8"
        ) as fd:
            fd.write("".join(eval_result))

    adam_filter(FLAGS.output_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
