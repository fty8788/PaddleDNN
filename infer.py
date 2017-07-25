#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017-06-18
@author: yitengfei

'''
import argparse

import paddle.v2 as paddle
import reader
from utils import TaskType, load_dic, logger, ModelType, ModelArch, display_args
from api import Inferer

parser = argparse.ArgumentParser(description="PaddlePaddle DNN model")

parser.add_argument(
    '-d',
    '--data_path',
    type=str,
    required=True,
    help="path of training dataset")
parser.add_argument(
    '-m',
    '--model_path',
    type=str,
    required=False,
    help="path of model parameters file")
parser.add_argument(
    '-o',
    '--prediction_output_path',
    type=str,
    required=True,
    help="path to output the prediction")
parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    default=100,
    help="size of mini-batch (default:100)")
parser.add_argument(
    '-y',
    '--model_type',
    type=int,
    required=True,
    default=ModelType.CLASSIFICATION_MODE,
    help=
    "model type, %d for classification, %d for regression (default: classification)"
    % (ModelType.CLASSIFICATION_MODE, ModelType.REGRESSION_MODE))
parser.add_argument(
    '-f',
    '--feature_dim',
    type=int,
    required=True,
    default=800,
    help="dimention of feature, default is 800")
parser.add_argument(
    '--dnn_dims',
    type=str,
    default='256,128,64,32',
    help=
    "dimentions of dnn layers, default is '256,128,64,32', which means create a 4-layer dnn, demention of each layer is 256, 128, 64 and 32"
)
parser.add_argument(
    '-c',
    '--class_num',
    type=int,
    default=2,
    help="number of categories for classification task.")

# arguments check.
args = parser.parse_args()
args.model_type = ModelType(args.model_type)
if args.model_type.is_classification():
    assert args.class_num > 1, "--class_num should be set in classification task."


if __name__ == '__main__':
    inferer = Inferer(args.model_path,
            model_type=args.model_type,
            feature_dim=args.feature_dim,
            dnn_dims=args.dnn_dims)
    inferer.infer(args.data_path, args.prediction_output_path, args.model_type, args.feature_dim, args.batch_size)
