#!/bin/bash

PID="10"
if ps -p $PID > /dev/null; then
  echo "Process $PID is running"
else
  echo "Process $PID is not running"
fi

datasetname="food"

while true; do
  if ! ps -p $PID > /dev/null; then
    CUDA_VISIBLE_DEVICES=1 /usr/local/anaconda3/envs/torch/bin/python step0_generate_word_list.py --dataset $datasetname
    CUDA_VISIBLE_DEVICES=1 /usr/local/anaconda3/envs/torch/bin/python step1_generate_classname_embedding.py --dataset $datasetname --shots 0
    CUDA_VISIBLE_DEVICES=1 /usr/local/anaconda3/envs/torch/bin/python step2_get_quantification_list_for_each_class.py --dataset $datasetname --shots 0
    CUDA_VISIBLE_DEVICES=1 /usr/local/anaconda3/envs/torch/bin/python step3_generate_and_filter_codebooks.py --dataset $datasetname --shots 0 --number_of_images 
    break
  else
    echo "Process $PID is running"
    sleep 600
  fi
done
