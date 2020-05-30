#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:
  * @file: run_tf_ner.py
  * @time: 2020/05/29
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import pickle
import logging

import tensorflow as tf
from bert_base.bert import tokenization, modeling
from bert_base.train import conlleval

from utils.data_utils import NerProcessor
from utils.data_utils import filed_based_convert_examples_to_features, file_based_input_fn_builder
from utils.model_utils import adam_filter, load_global_step_from_checkpoint_dir
from model.get_model import model_fn_builder

logging.getLogger().setLevel(tf.compat.v1.logging.INFO)

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
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

# Other parameters

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


def get_tf_record_data(tmp_file, examples, label_list, tokenizer, is_training=True, drop_remainder=True):
    if not os.path.exists(tmp_file):
        filed_based_convert_examples_to_features(
            examples,
            label_list,
            FLAGS.max_seq_length,
            tokenizer,
            tmp_file,
            FLAGS.output_dir,
        )

    # 2.读取record 数据，组成batch
    record_input_fn = file_based_input_fn_builder(
        input_file=tmp_file,
        seq_length=FLAGS.max_seq_length,
        is_training=is_training,
        drop_remainder=drop_remainder,
    )
    return record_input_fn


def train(estimator, num_train_steps, train_input_fn, eval_input_fn):
    # train and eval togither
    # early stop hook
    early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name="loss",
        max_steps_without_decrease=num_train_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=FLAGS.save_checkpoints_steps,
    )
    # initHook = InitHook(checkpoint_dir=run_config.model_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=num_train_steps,
        hooks=[early_stopping_hook],
    )
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


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
    history_max_steps = load_global_step_from_checkpoint_dir(FLAGS.output_dir)

    processor = processors[FLAGS.task_name](FLAGS.data_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case
    )

    session_config = tf.compat.v1.ConfigProto(
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
        num_train_steps = history_max_steps + int(len(train_examples) * 1.0 / FLAGS.batch_size * FLAGS.num_train_epochs)
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

    label_list = processor.get_labels()
    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        args=FLAGS,
    )

    params = {"batch_size": FLAGS.batch_size}
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config,
        # warm_start_from=run_config.model_dir,
    )

    if FLAGS.do_train and FLAGS.do_eval:
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        # 2.读取record 数据，组成batch
        train_input_fn = get_tf_record_data(train_file, train_examples, label_list, tokenizer)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        eval_input_fn = get_tf_record_data(eval_file, eval_examples, label_list, tokenizer)

        train(estimator, num_train_steps, train_input_fn, eval_input_fn)

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(FLAGS.output_dir, "label2id.pkl"), "rb") as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        predict_input_fn = get_tf_record_data(predict_file, predict_examples, label_list, tokenizer, False, False)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

        def result_to_pair(writer):
            for predict_line, prediction in zip(predict_examples, result):
                idx = 0
                line = ""
                line_token = predict_line.words
                label_token = predict_line.labels
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

        eval_result = conlleval.return_report(output_predict_file)
        print("".join(eval_result))
        # 写结果到文件中
        with codecs.open(os.path.join(FLAGS.output_dir, "predict_score.txt"), "a", encoding="utf-8") as fd:
            fd.write("".join(eval_result))

    adam_filter(FLAGS.output_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
