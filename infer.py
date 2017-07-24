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
    default=0,
    help="number of categories for classification task.")

# arguments check.
args = parser.parse_args()
args.model_type = ModelType(args.model_type)
if args.model_type.is_classification():
    assert args.class_num > 1, "--class_num should be set in classification task."

feature_dim = args.feature_dim
layer_dims = [int(i) for i in args.dnn_dims.split(',')]

paddle.init(use_gpu=False, trainer_count=1)

def create_dnn(sent_vec):
    # if more than three layers, than a fc layer will be added.
    if len(layer_dims) > 1:
        _input_layer = sent_vec
        for id, dim in enumerate(layer_dims):
            name = "fc_%d_%d" % (id, dim)
            logger.info("create fc layer [%s] which dimention is %d" %
                            (name, dim))
            fc = paddle.layer.fc(
                    name=name,
                    input=_input_layer,
                    size=dim,
                    act=paddle.activation.Relu())
            _input_layer = fc
    return _input_layer


class Inferer(object):
    def __init__(self, param_path):
        logger.info("create DNN model")

        # network config
        input_layer = paddle.layer.data(name='input_layer', type=paddle.data_type.dense_vector(feature_dim))
        dnn = create_dnn(input_layer)
        prediction = None
        label = None
        cost = None
        if args.model_type.is_classification():
            prediction = paddle.layer.fc(input=dnn, size=args.class_num, act=paddle.activation.Softmax())
            label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(args.class_num))
            cost = paddle.layer.classification_cost(input=prediction, label=label)
        elif args.model_type.is_regression():
            prediction = paddle.layer.fc(input=dnn, size=1, act=paddle.activation.Linear())
            label = paddle.layer.data(name='label', type=paddle.data_type.dense_vector(1))
            cost = paddle.layer.mse_cost(input=prediction, label=label)

        # load parameter
        logger.info("load model parameters from %s" % param_path)
        self.parameters = paddle.parameters.Parameters.from_tar(
                open(param_path, 'r'))
        self.inferer = paddle.inference.Inference(
                output_layer=prediction, parameters=self.parameters)

    def infer(self, data_path):
        logger.info("infer data...")

        infer_reader = reader.test(data_path,
                                            feature_dim+1,
                                            args.model_type.is_classification())
        infer_batch = paddle.batch(reader.all(data_path,
                                            feature_dim+1,
                                            args.model_type.is_classification()),
                            batch_size=args.batch_size)

        logger.warning('write predictions to %s' % args.prediction_output_path)
        output_f = open(args.prediction_output_path, 'w')

        batch = []
        #for item in infer_reader():
        #    batch.append([item[0]])
        for id, batch in enumerate(infer_batch()):
            res = self.inferer.infer(input=batch)
            predictions = [' '.join(map(str, x)) for x in res]
            assert len(batch) == len(
                    predictions), "predict error, %d inputs, but %d predictions" % (
                            len(batch), len(predictions))
            output_f.write('\n'.join(map(str, predictions)) + '\n')
            batch = []


if __name__ == '__main__':
    inferer = Inferer(args.model_path)
    inferer.infer(args.data_path)
