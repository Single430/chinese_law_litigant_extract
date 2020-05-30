#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:
  * @file: model_utils.py
  * @time: 2020/05/29
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs

import tensorflow as tf
from tensorflow.python.training import training
from tensorflow.python.framework import ops


def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, "checkpoint")):
        tf.logging.info(
            "checkpoint file not exits:".format(os.path.join(model_path, "checkpoint"))
        )
        return None
    last = None
    with codecs.open(
        os.path.join(model_path, "checkpoint"), "r", encoding="utf-8"
    ) as fd:
        for line in fd:
            line = line.strip().split(":")
            if len(line) != 2:
                continue
            if line[0] == "model_checkpoint_path":
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path:
    :return:
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(
        os.path.join(model_path, last_name + ".meta")
    )
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if "adam_v" not in var.name and "adam_m" not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, "model.ckpt"))


def load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


class InitHook(tf.estimator.SessionRunHook):
    """initializes model from a checkpoint_path
    FLAGS:
        modelPath: full path to checkpoint
    """

    def __init__(self, checkpoint_dir):
        self.modelPath = checkpoint_dir
        self.initialized = False

    def begin(self):
        """
        Restore encoder parameters if a pre-trained encoder model is available and we haven't trained previously
        """
        if not self.initialized:
            checkpoint = tf.train.latest_checkpoint(self.modelPath)
            if checkpoint is None:
                tf.logging.info(
                    "No pre-trained model is available, training from scratch."
                )
            else:
                tf.logging.info(
                    "Pre-trained model {0} found in {1} - warmstarting.".format(
                        checkpoint, self.modelPath
                    )
                )
                tf.train.warm_start(checkpoint)
            self.initialized = True

