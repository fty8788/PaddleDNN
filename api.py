#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017-06-18
@author: yitengfei

'''
import paddle.v2 as paddle
import reader
from utils import TaskType, load_dic, logger, ModelType, ModelArch, display_args

def create_dnn(sent_vec, layer_dims):
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
    def __init__(self, param_path,
            model_type=ModelType(ModelType.CLASSIFICATION_MODE),
            class_num=2,
            feature_dim=800,
            dnn_dims='256,128,64,32'):
        logger.info("create DNN model")

        paddle.init(use_gpu=False, trainer_count=1)

        # network config
        input_layer = paddle.layer.data(name='input_layer', type=paddle.data_type.dense_vector(feature_dim))
        layer_dims = [int(i) for i in dnn_dims.split(',')]
        dnn = create_dnn(input_layer, layer_dims)
        prediction = None
        label = None
        cost = None
        if model_type.is_classification():
            prediction = paddle.layer.fc(input=dnn, size=class_num, act=paddle.activation.Softmax())
            label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(class_num))
            cost = paddle.layer.classification_cost(input=prediction, label=label)
        elif model_type.is_regression():
            prediction = paddle.layer.fc(input=dnn, size=1, act=paddle.activation.Linear())
            label = paddle.layer.data(name='label', type=paddle.data_type.dense_vector(1))
            cost = paddle.layer.mse_cost(input=prediction, label=label)

        # load parameter
        logger.info("load model parameters from %s" % param_path)
        self.parameters = paddle.parameters.Parameters.from_tar(
                open(param_path, 'r'))
        self.inferer = paddle.inference.Inference(
                output_layer=prediction, parameters=self.parameters)

    def infer(self, data_path, output_path,
            model_type=ModelType(ModelType.CLASSIFICATION_MODE),
            feature_dim=800,
            batch_size=100):
        logger.info("infer data...")

        #infer_reader = reader.test(data_path,
        #                                    feature_dim+1,
        #                                    model_type.is_classification())
        infer_batch = paddle.batch(reader.test(data_path,
                                            feature_dim+1,
                                            model_type.is_classification()),
                            batch_size=batch_size)

        logger.warning('write predictions to %s' % output_path)
        output_f = open(output_path, 'w')

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

    def infer_one(self, one, seperator=" "):
        logger.info("infer one...")

        batch = []
        one_input = one.split(seperator)
        batch.append([one_input])
        res = self.inferer.infer(input=batch)
        predictions = [' '.join(map(str, x)) for x in res]
        assert len(batch) == len(
                    predictions), "predict error, %d inputs, but %d predictions" % (
                            len(batch), len(predictions))
        logger.info('\n'.join(map(str, predictions)) + '\n')
        return predictions


if __name__ == '__main__':
    inferer = Inferer(args.model_path)
    inferer.infer(args.data_path, args.prediction_output_path, args.batch_size)
    one = "0.1633836 -0.3711518 -0.0259373 0.1322735 0.0876563 0.2604049 0.0125503 0.2634946 0.1022274 0.6153814 -0.1526374 -0.1024764 0.1961514 -0.2821807 -0.25327 -0.5000692 0.2685974 -0.5079111 -0.1186484 -0.7617657 0.7773207 -0.1684354 0.7884739 -0.0648165 -0.272902 -0.2471674 0.5539562 0.3617943 0.260433 -0.071835 -0.3589811 0.3097286 -0.3026498 0.2765978 -0.3272495 0.6800184 -0.2231477 0.2088786 -0.0150541 0.1090587 -0.4599472 -0.363955 0.0969468 -0.0977496 0.2036859 0.1080659 0.0990615 0.3070166 0.0316247 0.2655011 0.1714281 -0.2027331 0.2651861 -0.3298607 0.1064283 -0.2965369 0.3558804 0.5468868 0.3141988 -0.1171853 0.5206051 -0.2866887 0.1220448 -0.2326679 0.2364277 -0.1314874 -0.2517253 0.0260689 -0.2098157 -0.2741065 0.057636 0.1133775 -0.2352439 -0.1175847 0.4430246 -0.2694384 0.1020924 0.4553988 -0.1107943 0.1271083 -0.3202396 0.4015112 0.3783104 0.0147211 0.1203903 0.075937 0.007773 -0.3967534 0.6905375 0.0338501 0.0166612 -0.3402868 -0.4343578 -0.1815295 -0.5327111 0.4634258 -0.1649225 0.1153338 0.6397334 0.1675995 -0.0964657 -0.1374637 -0.125555 0.5183492 0.0153004 0.0863708 0.2970177 -0.0210646 -0.1460709 -0.3768268 -0.5529064 0.1408289 0.2891883 -0.2618959 -0.1311019 0.0164649 0.0041663 -0.1177589 0.1182694 0.1202841 0.1119099 0.2013878 0.0272903 0.214807 0.3824635 0.1882141 0.3972285 0.6790356 -0.8182583 -0.113595 0.1418323 0.3215305 -0.2315272 -0.4028556 0.01774 -0.1015072 0.0838364 0.1666582 0.0079259 0.1693448 0.5262812 -0.0107787 -0.335942 0.0480012 -0.1199989 0.4173049 -0.0198625 -0.0240674 0.3827431 -0.1015504 0.4077672 0.3689588 -0.3646717 0.7276221 -0.7302616 -0.3167725 0.3436458 0.290301 0.0049577 0.0985046 0.0648259 0.2373409 -0.0619335 0.3138082 -0.0278676 -0.6297871 -0.2506534 -0.0969237 0.6386227 0.0557263 -0.5457233 0.1049742 -0.1511433 0.1064275 0.1114698 -0.3672346 -0.0723686 -0.178739 -0.0027702 0.5182293 0.0244293 -0.3124585 0.159318 0.1336769 0.1235747 -0.1543294 -0.602284 0.0311669 0.0580838 -0.3888577 -0.1304036 0.2008377 0.0003844 -0.02319 0.4697954 -0.1977844 -0.3262251 0.1321449 0.0528502 0.035388 0.3040323 -0.3563201 -0.1265566 0.0724993 0.2750453 -0.4074266 0.3160345 -0.1499143 -0.0319494 -0.359521 -0.5188788 0.4124469 -0.3726021 -0.0350876 -0.0799021 -0.3796259 0.2504503 0.1023042 0.2508772 -0.1617834 0.2522005 -0.3228006 0.1316565 -0.473707 0.2678094 -0.0280072 0.1095562 -0.4825032 0.4629398 0.1089284 0.6296628 0.2779462 0.4175847 0.0024762 -0.1213177 0.3693239 -0.5037153 -0.1596329 -0.5088571 -0.481088 0.1840263 -0.0939124 -0.2740663 0.1733588 -0.2104957 -0.3357643 0.2519626 -0.0556882 0.5832371 0.011006 0.7884186 0.0744374 -0.2388166 0.3944759 0.0021113 0.5527008 -0.1785755 0.1154866 0.0957654 0.1834947 -0.165236 0.304533 -0.054403 -0.1348931 0.363749 0.0960522 0.3621441 -0.1405579 0.0870802 -0.3682373 -0.0674789 -0.3418013 0.3342441 0.1553916 -0.117446 -0.178493 0.1578566 -0.1775379 -0.5251691 -0.0804979 0.0720055 0.0977989 -0.2101882 0.1277508 0.2976649 -0.0150983 0.3523222 0.0111039 -0.6375181 -0.542869 -0.3853287 0.3495397 -0.1589097 0.5518703 0.0107362 0.3440556 -0.4957424 -1.0828923 -0.063211 0.0210657 -0.1519969 0.044674 -0.0435032 0.3661898 0.1421809 0.3393811 0.1411019 -0.0227774 0.1157397 0.0516675 0.064918 0.2726687 -0.4250136 -0.6042527 -0.2130835 0.455336 0.3180404 0.5263634 -0.0969809 0.0506635 -0.2716642 0.0987069 -0.2756117 0.0707727 -0.4524571 -0.0914149 0.2515624 -0.04487 -0.2505645 -0.2145495 -0.5319554 0.3766065 -0.0486938 -0.361494 0.1590388 -0.376639 0.2146329 -0.3744871 0.5776039 0.4670003 0.2764826 -0.407357 0.2085321 -0.2850049 0.2684525 0.4040956 0.4043293 -0.3901492 -0.1490816 0.3495256 0.2900283 -0.1606391 -0.4043663 0.1772525 -0.309891 -0.218217 -0.1636019 0.1092689 0.0117914 0.2196625 0.3465425 -0.0410356 0.0792378 -0.0822546 0.6929122 -0.2373888 -0.4846336 -0.0297892 0.1097356 0.253486 -0.3591771 0.3459858 -0.1435835 -0.3852024 -0.1589915 0.1503151 -0.7564108 0.192663 0.4734719 -0.1476399 -0.1633819 0.009688 -0.2475886 -0.1332078 -0.1628105 0.2205151 -0.2718264 -0.1319524 -0.1910747 -0.0453656 -0.2443877 0.2971488 0.163519 -0.4047085 -0.0513989 0.2021605 -0.6941103 0.3137927 0.1553976 -0.0530794 0.206897 -0.478153888889 -0.0682238888889 0.0717098888889 0.015556 0.293199666667 0.224345111111 0.020274 0.0351653333333 0.900506666667 -0.213394444444 -0.216594111111 0.164350111111 -0.381836555556 -0.238356777778 -0.318359333333 0.287522333333 -0.778399777778 -0.153362111111 -0.835621222222 0.756318777778 -0.0181167777778 0.855504222222 0.0343552222222 -0.487989444444 -0.357545222222 0.557638555556 0.305464555556 0.235764888889 -0.0361277777778 -0.262194222222 0.480053 -0.337083888889 0.139039666667 -0.288510111111 0.879915111111 -0.286436444444 0.209699 -0.150958333333 0.0941391111111 -0.691569777778 -0.489905111111 -0.0407817777778 0.0106815555556 0.325969555556 0.31301 0.0623614444444 0.491647 0.275970444444 0.370647666667 0.235580444444 -0.255640777778 0.217363888889 -0.325248555556 -0.0473974444444 -0.403426333333 0.529476888889 0.65712 0.399659777778 -0.240615111111 0.730069555556 -0.358948333333 0.45157 -0.343129 -0.203938 0.0532913333333 -0.254715444444 0.131655777778 -0.0483266666667 -0.254012333333 -0.100632444444 0.0152906666667 -0.309758777778 -0.0910845555556 0.534363555556 -0.356948444444 0.0193826666667 0.271445555556 -0.241000111111 0.0639004444444 -0.0915398888889 0.374109333333 0.366449555556 -0.0883661111111 0.111226333333 -0.168131222222 -0.0994382222222 -0.495902 0.713122666667 0.344688888889 0.291190333333 -0.307570222222 -0.556263888889 0.0305178888889 -0.488152555556 0.300473333333 0.0419976666667 -0.0378092222222 0.575484555556 0.0575901111111 -0.224294222222 -0.0471397777778 -0.197265555556 0.368827888889 0.134167111111 0.0953683333333 0.310697333333 -0.0830784444444 -0.221690111111 -0.392691777778 -0.429288222222 0.00777855555556 0.297524222222 -0.459513111111 0.0185825555556 0.0769575555556 -0.118272 -0.260614 0.178984333333 -0.150355 0.121532222222 0.183559111111 -0.02123 0.265731 0.102953 0.0170757777778 0.387936111111 0.515406333333 -0.870317555556 -0.323193666667 0.226084666667 0.358388666667 -0.222842333333 -0.199813666667 0.258905777778 -0.150411222222 0.174968333333 0.440314777778 -0.301267888889 0.294042555556 0.595954444444 0.0952657777778 -0.435636666667 0.119665444444 0.0345326666667 0.291593555556 0.168157 0.0801195555556 0.583718111111 -0.108392111111 0.328421222222 0.408264777778 -0.366603666667 0.898015555556 -0.920428333333 -0.476226444444 0.377739333333 0.205842444444 0.0816802222222 0.0106365555556 0.174413 0.303983888889 0.00445677777778 0.0239425555556 0.270572 -0.568730444444 -0.451673777778 -0.129531 0.445537444444 -0.130590555556 -0.753032 -0.0807845555556 -0.215194666667 -0.0621854444444 -0.0274814444444 -0.552506555556 0.041384 -0.0163643333333 0.221578444444 0.552270888889 0.0533272222222 -0.340171111111 0.285759333333 0.147773666667 -0.237706 -0.097961 -0.501869111111 0.0755723333333 -0.203459888889 -0.402264333333 -0.313479333333 0.261866111111 0.141997888889 -0.147522444444 0.648474111111 -0.386079444444 -0.333201 0.340103555556 0.139706 0.14472 0.407734666667 -0.425921777778 -0.273939666667 0.149806444444 0.456010555556 -0.484408888889 0.336181666667 -0.277438333333 -0.0309976666667 -0.429444666667 -0.698581111111 0.483850666667 -0.203677111111 0.00506877777778 -0.0710226666667 -0.272443666667 0.297918222222 0.243016 0.0263394444444 -0.275514111111 0.580564444444 -0.312515777778 0.220650666667 -0.658505111111 0.357544222222 -0.207161666667 -0.0597444444444 -0.366424666667 0.190768777778 0.0781171111111 0.705602888889 0.189163666667 0.418086555556 0.0907463333333 -0.242003111111 0.381175111111 -0.393715 0.0793988888889 -0.542362666667 -0.566001666667 -0.131170777778 0.0727004444444 -0.222974666667 0.118656 -0.196911444444 -0.246672777778 0.131646 -0.135813222222 0.412395555556 0.0725027777778 0.711838111111 -0.00881955555556 -0.192839 0.259014222222 -0.260198222222 0.53374 -0.0947415555556 0.0823513333333 0.226808222222 0.108043111111 -0.228718666667 0.351285111111 0.122167 -0.333000444444 0.136452333333 -0.0181927777778 0.243465 -0.140750888889 0.230082888889 -0.431106333333 -0.0575502222222 -0.343320222222 0.261390333333 0.287793888889 -0.151048666667 -0.129319777778 0.0109221111111 -0.348347555556 -0.605288555556 -0.160375222222 0.177056333333 -0.106656333333 -0.145253222222 0.116827555556 0.265559555556 -0.273026888889 0.556710333333 -0.105364888889 -0.808669777778 -0.654979 -0.316566333333 0.381603777778 -0.303446 0.715533888889 -0.0197554444444 0.3201 -0.538170555556 -1.26489522222 -0.146542888889 0.0317558888889 -0.185645 0.0990261111111 0.020212 0.430082666667 0.0154182222222 0.577024 0.193222777778 0.119606666667 -0.268170222222 0.297489555556 0.142722111111 0.437825888889 -0.573574222222 -0.665224666667 -0.106222111111 0.560025222222 0.513398666667 0.54764 -0.00381522222222 -0.0609901111111 -0.378634111111 -0.0207386666667 -0.279441666667 0.156435 -0.409218777778 -0.0980733333333 0.349754333333 -0.00524466666667 -0.152626777778 -0.220258111111 -0.518838333333 0.318287555556 0.112159555556 -0.282657222222 0.316473444444 -0.399052444444 0.280785777778 -0.235576777778 0.585335333333 0.557532444444 0.458915666667 -0.710178777778 0.251745222222 -0.529776555556 0.234848 0.201047222222 0.452784222222 -0.262632222222 -0.387993222222 0.293166 0.231737666667 -0.238721888889 -0.357456 0.266999555556 -0.542844666667 -0.452749 -0.125204888889 0.304601111111 -0.0806068888889 0.254233777778 0.420704444444 -0.0964566666667 0.117716111111 0.0109984444444 0.630107888889 -0.207532222222 -0.385798222222 -0.0593314444444 0.227177777778 0.0273067777778 -0.301845777778 0.372739666667 -0.124322111111 -0.300995777778 -0.216158888889 0.210973555556 -0.677621111111 0.200140333333 0.620540111111 -0.357263666667 -0.0563473333333 0.0377947777778 -0.309078888889 -0.311370444444 -0.284171222222 0.0460415555556 -0.202294111111 0.0844605555556 -0.140338333333 -0.0721196666667 -0.549203444444 0.241420222222 0.223593333333 -0.671648555556 0.133446666667 0.353629333333 -0.600478666667 0.440234666667 0.115513222222 -0.166708888889"
    print inferer.infer_one(one)
