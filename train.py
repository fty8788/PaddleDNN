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
    required=False,
    help="path of training dataset")
parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    default=100,
    help="size of mini-batch (default:100)")
parser.add_argument(
    '-p',
    '--num_passes',
    type=int,
    default=50,
    help="number of passes to run(default:50)")
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
parser.add_argument(
    '--num_workers', type=int, default=1, help="num worker threads, default 1")
parser.add_argument(
    '--use_gpu',
    type=bool,
    default=False,
    help="whether to use GPU devices (default: False)")

# arguments check.
args = parser.parse_args()
args.model_type = ModelType(args.model_type)
if args.model_type.is_classification():
    assert args.class_num > 1, "--class_num should be set in classification task."

feature_dim = args.feature_dim
layer_dims = [int(i) for i in args.dnn_dims.split(',')]

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

def train(data_path=None,
          model_type=ModelType.create_classification(),
          batch_size=100,
          num_passes=50,
          class_num=None,
          num_workers=1,
          use_gpu=False):
    '''
    Train the DNN.
    '''
    paddle.init(use_gpu=use_gpu, trainer_count=num_workers)

    # network config
    input_layer = paddle.layer.data(name='input_layer', type=paddle.data_type.dense_vector(feature_dim))
    dnn = create_dnn(input_layer)
    prediction = None
    label = None
    cost = None
    if args.model_type.is_classification():
        prediction = paddle.layer.fc(input=dnn, size=class_num, act=paddle.activation.Softmax())
        label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(class_num))
        cost = paddle.layer.classification_cost(input=prediction, label=label)
    elif args.model_type.is_regression():
        prediction = paddle.layer.fc(input=dnn, size=1, act=paddle.activation.Linear())
        label = paddle.layer.data(name='label', type=paddle.data_type.dense_vector(1))
        cost = paddle.layer.mse_cost(input=prediction, label=label)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0)

    trainer = paddle.trainer.SGD(
        cost=cost, 
        extra_layers=paddle.evaluator.auc(input=prediction, label=label),
        parameters=parameters, update_equation=optimizer)

    feeding = {'input_layer': 0, 'label': 1}

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(reader.test(data_path,
                                            feature_dim+1,
                                            args.model_type.is_classification()),
                            batch_size=batch_size),
                feeding=feeding)
            print "Test %d, Cost %f, %s" % (event.pass_id, result.cost, result.metrics)

    # training
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(reader.train(data_path,
                                            feature_dim+1,
                                            args.model_type.is_classification()),
                    buf_size=batch_size*10),
            batch_size=batch_size),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=num_passes)


if __name__ == '__main__':
    display_args(args)
    train(
        data_path=args.data_path,
        model_type=ModelType(args.model_type),
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        class_num=args.class_num,
        num_workers=args.num_workers,
        use_gpu=args.use_gpu)
