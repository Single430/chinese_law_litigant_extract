#!/usr/bin/env bash

PREBERT_PATH=/run/media/zbl/works/python/model-data/bert_hinese_L-12_H-768_A-12/tf1.0

python run_tf_ner.py -data_dir ./data \
-output_dir ./output-ner-law \
-task_name ner \
-do_train \
-do_eval \
-do_predict \
-init_checkpoint ${PREBERT_PATH}/bert_model.ckpt \
-bert_config_file ${PREBERT_PATH}/bert_config.json \
-vocab_file ${PREBERT_PATH}/vocab.txt -max_seq_length 512 -batch_size 4
