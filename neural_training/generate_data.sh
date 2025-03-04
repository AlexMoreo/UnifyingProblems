#!/bin/bash

for dataset in yelp imdb ag_news sst2 ; do
  for model in google-bert/bert-base-uncased distilbert/distilbert-base-uncased FacebookAI/roberta-base ; do
    CUDA_VISIBLE_DEVICES=1 python train_and_embed.py --nepochs 5 --model_name $model --dataset_name $dataset --val_size 5000
  done
done



