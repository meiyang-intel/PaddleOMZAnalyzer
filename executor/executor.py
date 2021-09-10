import argparse
import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
import csv

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import paddle
import paddle.inference as paddle_infer
# from paddle.inference import Config
# from paddle.inference import create_predictor

from abc import ABCMeta, abstractmethod

from utils import is_float_tensor, randtool, get_tensor_dtype
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork, StatusCode
from collections import namedtuple
#PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdparams compare_result')
PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdmodel compare_result')

class Executor(object):
    __metaclass__ = ABCMeta

    def __init__(self, pdmodel:str):
        self.__pdmodel__ = pdmodel

        self.inputs_info = dict()  # shape, dtype of inputs
        self.outputs_info = dict() # shape, dtype of outputs
        self.inference_results = dict() # note: make inference results of each executor be dict for comparision.

    def get_inputs_info(self):
        """
        get info like shape, dtype of inputs.
        """
        return self.inputs_info

    def get_outputs_info(self):
        """
        get info like shape, dtype of outputs.
        """
        return self.outputs_info

    def generate_inputs(self, batch_size):
        """
        generate input tensors.
        note: This may going to be shared by PDPD and OV executors for accuracy comparision.
        """
        inputs_info = self.get_inputs_info()

        # random generated inputs
        test_inputs = dict()
        for k, v in inputs_info.items(): # input_name: (shape, dtype)
            shape = list(v[0])
            shape[0] = batch_size

            # like fast-scnn without spatial resolution specified.
            shape = [shape[i] if shape[i] != -1 else randtool("int", 1, 800, None) for i in range(len(shape))]
            print(shape)
            if is_float_tensor(v[1]):
                test_data = randtool("float", -1, 1, shape).astype(get_tensor_dtype(v[1]))
            else:
                test_data = randtool("int", -1, 1, shape).astype(get_tensor_dtype(v[1]))

            test_inputs[k] = test_data
        return test_inputs

    def get_inference_results(self):
        """
        get info like shape, dtype of inputs.
        """
        return self.inference_results

    @abstractmethod
    def inference(self, inputs:dict, warmup=0, repeats=1):
        """
        Inferencing
        inputs [in] a dict of input name and tensor.
        """
        pass

    def run(self, test_inputs, warmup=0, repeats=1):
        """
        Generate input randomly, and inferencing
        """

        # inference
        self.inference(test_inputs, warmup, repeats)

class PaddleExecutor(Executor):
    """
    inputs of Paddle are dict.
    outputs possibly either -
    1. list of numpy.ndarray/numpy.generic, or LodTensor, depends on 'return_numpy',
    or,
    2. numpy.ndarray.
    """
    def __init__(self, pdmodel):
        super().__init__(pdmodel)

        paddle.enable_static()
        # Paddle-private
        self.inference_program = None
        self.feed_target_names = None
        self.fetch_targets = None
        self.exe = None
        self.output = None

        # load model
        self.exe = paddle.static.Executor(paddle.CPUPlace())
        [self.inference_program, self.feed_target_names, self.fetch_targets] = paddle.static.load_inference_model(
                            os.path.splitext(self.__pdmodel__)[0], # pdmodel prefix
                            self.exe)

        print(self.feed_target_names, self.fetch_targets)

        # 输出计算图所有input结点信息
        for ipt in self.feed_target_names:
            var=self.inference_program.global_block().var(ipt)
            print('model input name {} with shape {}, dtype {}'.format(ipt, var.shape, var.dtype))
            self.inputs_info[ipt] = (var.shape, var.dtype)

        # workaround for yolo and ppyolo,
        # which requires return_numpy False, else exception prompts from Paddle.
        self.return_numpy = True
        # 输出计算图所有结点信息
        for i, op in enumerate(self.inference_program.blocks[0].ops):
            if op.type in { 'matrix_nms', 'multiclass_nms3', 'multiclass_nms2', 'multiclass_nms' }:
                self.return_numpy = False

    def inference(self, inputs:dict, warmup=0, repeats=1):
        #warm up
        for i in range(warmup):
            inference_results = self.exe.run(self.inference_program, feed=inputs, fetch_list=self.fetch_targets, return_numpy = self.return_numpy)

        #repeats
        t1 = time.time()
        for i in range(repeats):
            inference_results = self.exe.run(self.inference_program, feed=inputs, fetch_list=self.fetch_targets, return_numpy = self.return_numpy)
        t2 = time.time()
        ms = (t2 - t1) * 1000.0 / repeats
        print("PaddleExecutor Inference: {} ms per batch image".format(ms))

        # debug info
        print("PaddleExecutor inference results type {}, len {}, type of element {}".format(type(inference_results), len(inference_results), type(inference_results[0])))

        # if self.return_numpy is False: # workaround for yolo and ppyolo.. no need care, as compare() will handle it.
        #     inference_results = [np.array(res) for res in inference_results]

        # convert inference results to dict, in order to compare to openvino results.
        for i in range(len(self.fetch_targets)):
            self.inference_results[self.fetch_targets[i].name] = inference_results[i]

class PaddlePredictorExecutor(Executor):
    """
    inputs and outputs of PaddlePredictor are dict.
    """
    def __init__(self, pdmodel, pdiparams):
        super().__init__(pdmodel)
        self.pdiparams = pdiparams
        self.config = paddle_infer.Config(self.__pdmodel__, self.pdiparams)
        self.config.enable_mkldnn()
        self.predictor = paddle_infer.create_predictor(self.config)

    def inference(self, inputs:dict, warmup=0, repeats=1):
        input_names = self.predictor.get_input_names()
        for input_name in input_names:
            input_handle = self.predictor.get_input_handle(input_name)
            input_handle.reshape(inputs[input_name].shape)
            input_handle.copy_from_cpu(inputs[input_name])
        #warmup
        for i in range(warmup):
            self.predictor.run()

        #repeats
        t1 = time.time()
        for i in range(repeats):
            self.predictor.run()
        t2 = time.time()
        ms = (t2 - t1) * 1000.0 / repeats
        print("PaddlePredictorExecutor Inference: {} ms per batch image".format(ms))

        output_names = self.predictor.get_output_names()
        for output_name in output_names:
            output_handle = self.predictor.get_output_handle(output_name)
            self.inference_results[output_name] = output_handle.copy_to_cpu()


class OpenvinoExecutor(Executor):
    """
    inputs and outputs of OpenVINO are dict.
    """
    def __init__(self, pdmodel, number_infer_requests: int = None, api_type: str = 'sync'):
        super().__init__(pdmodel)
        self.ie = IECore()
        self.api_type = api_type
        self.nireq = number_infer_requests
        self.net = self.ie.read_network(self.__pdmodel__)

    def inference(self, inputs:dict, warmup=0, repeats=1):
        # reshape the net work
        reshape_dict = {}
        for item in list(self.net.input_info.items()):
            reshape_dict[item[0]]=inputs[item[0]].shape
        self.net.reshape(reshape_dict)

        # create executable network
        self.exec_net = self.ie.load_network(self.net, 'CPU', num_requests=1 if self.api_type == 'sync' else self.nireq or 0) # device
        self.nireq = len(self.exec_net.requests)
        self.niter = repeats
        print("OpenvinoExecutor final repeats:", self.nireq)
        assert isinstance(self.exec_net, ExecutableNetwork)

        #warmup
        for i in range(warmup):
            infer_request = self.exec_net.requests[0]
            if self.api_type == 'sync':
                infer_request.infer(inputs)
            else:
                infer_request.async_infer(inputs)
                status = infer_request.wait()
                if status != StatusCode.OK:
                    print("Wait for request is failed with status code {status}!")
                    return

        #inference
        infer_requests = self.exec_net.requests
        iteration = 0
        t1 = time.time()

        times = []
        in_fly = set()
        while (self.niter and iteration < self.niter):
            if self.api_type == 'sync':
                self.inference_results = self.exec_net.infer(inputs)
            else:
                infer_request_id = self.exec_net.get_idle_request_id()
                if infer_request_id < 0:
                    status = self.exec_net.wait(num_requests=1)
                    if status != StatusCode.OK:
                        raise Exception("Wait for idle request failed!")
                    infer_request_id = self.exec_net.get_idle_request_id()
                    if infer_request_id < 0:
                        raise Exception("Invalid request id!")
                in_fly.add(infer_request_id)
                infer_requests[infer_request_id].async_infer(inputs)
            iteration += 1

        # wait the latest inference executions
        status = self.exec_net.wait()
        if status != StatusCode.OK:
            raise Exception(f"Wait for all requests is failed with status code {status}!")

        t2 = time.time()
        ms = (t2 - t1) * 1000.0 / repeats
        print("OpenvinoExecutor Inference: {} ms per batch image".format(ms))

        #assert(type(self.inference_results) == dict)

def compare(result, expect, delta=1e-6, rtol=1e-6):
    """
    比较函数
    :param result: openvino
    :param expect: paddlepaddle
    :param delta: 误差值
    :return: True or False
    """
    res = True
    if type(result) == np.ndarray:
        expect = np.array(expect)
        strDtype = str(result.dtype)
        if strDtype.startswith('int') or strDtype.startswith('uint'):
            rtol = 0
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        if res is False:
            print(result, expect)
        return res
    elif type(result) == list:
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                res = compare(result[i], expect[i], delta, rtol)
            else:
                res = compare(result[i], expect[i].numpy(), delta, rtol)

            if res is not True:
                return False
    else:
        print("Compare Format Error.")
        return False

    return True

def inference_and_compare(model_file, test_mode, batch_size, warmup=0, repeats=1):
    if os.path.exists(model_file):
        #inference
        ## paddle inference
        pdpd_executor = PaddleExecutor(model_file)
        test_inputs = pdpd_executor.generate_inputs(batch_size)

        ## openvino inference
        ov_executor = OpenvinoExecutor(model_file)
        if test_mode == 'performance':
            ov_executor = OpenvinoExecutor(model_file, 10, 'async')
        ov_executor.run(test_inputs, warmup, repeats)
        ov_result = ov_executor.get_inference_results()

        ## create pdiparams file path
        dir_name = os.path.dirname(model_file)
        config_base = os.path.basename(model_file)
        config_base = os.path.splitext(config_base)[0]
        model_params_file = dir_name + '/' + config_base + '.pdiparams'

        ## paddle predictor inference
        pdpd_predict_executor = PaddlePredictorExecutor(str(model_file), model_params_file)
        pdpd_predict_executor.run(test_inputs, warmup, repeats)
        pdpd_predict_result = pdpd_predict_executor.get_inference_results()
        print(type(pdpd_predict_result))

        # compare openvino result and paddle result
        res = True
        if test_mode == 'accuracy':
            ov_result = [ov_result[k] for k in sorted(ov_result)]
            pdpd_predict_result = [pdpd_predict_result[k] for k in sorted(pdpd_predict_result)]
            res = compare(ov_result, pdpd_predict_result)
        return res
    else:
        return False

def loop_inference_and_compare(model_category: str, test_mode: str,  batch_size=1, warmup=0, repeats=1):
    """
    循环推理和比较函数
    :param model_category: 支持的类别,detection,classify,segmentation
    :return: True or False
    """
    # judge the params range
    model_category_list = ['detection', 'classify', 'segmentation']
    if model_category not in model_category_list:
        print('No support this {} model category'.format(model_category))
        return False

    # get csv_file and exported_path by model_category
    csv_file = ''
    model_category_path =''
    result_file =''

    if model_category == 'detection':
        csv_file = os.path.join(__dir__, '../downloader/paddledet_filtered.csv')
        model_category_path =os.path.join(__dir__, '../exporter/paddledet')
        result_file = os.path.join(__dir__, './paddledet_result.csv')
    elif model_category == 'classify':
        csv_file = os.path.join(__dir__, '../downloader/paddleclas_full.csv')
        model_category_path =os.path.join(__dir__, '../exporter/paddleclas')
        result_file = os.path.join(__dir__, './paddleclas_result.csv')
    elif model_category == 'segmentation':
        csv_file = os.path.join(__dir__, '../downloader/paddleseg_full.csv')
        model_category_path =os.path.join(__dir__, '../exporter/paddleseg')
        result_file = os.path.join(__dir__, './paddleseg_result.csv')

    if not os.path.exists(csv_file) or not os.path.exists(model_category_path):
        print('Have no this {} directory or {} not exist.'.format(model_category_path, csv_file))
        return False

    compare_result_list = []
    # loop inference and compare
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]

            exported_path = os.path.abspath(os.path.join(model_category_path, config_base))

            if os.path.exists(exported_path):
                p = Path(exported_path)
                pdmodel = p.glob('**/*.pdmodel')

                models = []
                for path in pdmodel:
                   models.append(path)
                if len(models) != 1:
                   print('{} have no *.pdmodel or have multi files exist'.format(exported_path))
                   continue

                res = inference_and_compare(models[0], test_mode, batch_size, warmup, repeats)
                compare_result_list.append(PDModelInfo(row[0], row[1], models[0], "Equal" if res else 'No Equal'))
            else:
                print('Have no this {} directory'.format(exported_path))

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(compare_result_list)

def main():
    args = parse_args()
    if args.mode == 'all':
        loop_inference_and_compare('classify', args.test_mode, args.batch_size, args.warmup, args.repeats)
        loop_inference_and_compare('detection', args.test_mode, args.batch_size, args.warmup, args.repeats)
        loop_inference_and_compare('segmentation', args.test_mode, args.batch_size, args.warmup, args.repeats)
    elif args.mode == 'single':
        res = inference_and_compare(args.model_file, args.test_mode, args.batch_size, args.warmup, args.repeats)
        if res:
            print("Equal")
        else:
            print("Not Equal")
    elif args.mode == 'category':
        if args.category == 'classify':
            loop_inference_and_compare('classify', args.test_mode, args.batch_size, args.warmup, args.repeats)
        elif args.category == 'detection':
            loop_inference_and_compare('detection', args.test_mode, args.batch_size, args.warmup, args.repeats)
        elif args.category == 'segmentation':
            loop_inference_and_compare('segmentation', args.test_mode, args.batch_size, args.warmup, args.repeats)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    det_category_path =os.path.abspath(os.path.join(__dir__, '../exporter/paddledet'))
    clas_category_path =os.path.abspath(os.path.join(__dir__, '../exporter/paddleclas'))
    seg_category_path =os.path.abspath(os.path.join(__dir__, '../exporter/paddleseg'))
    parser.add_argument("--test_mode", type=str, default='accuracy', choices=['accuracy', 'performance'], help="accuracy: compare paddle and openvino result after inference\nperformance: do not compare the result after inference\n(default:accuracy)")
    parser.add_argument("--mode", type=str, default='single', choices=['single', 'category', 'all'], help="single: only execute the single model_file you specify\ncategory: execute all models belong to the category you specify\nall: execute all models in all categorys\n(default:single)")
    parser.add_argument("--model_file", type=str, default=os.path.join(__dir__, '../exporter/paddleclas/MobileNetV1/inference.pdmodel'), help="model filename, only effect on single mode")
    parser.add_argument("--category", type=str, default='classify', choices=['classify', 'detection', 'segmentation'], help="classify: execute all models in {}\ndetection: execute all models in {}\nsegmentation: execute all models in {}\n(default:classify)\nonly effect on category mode".format(clas_category_path, det_category_path,  seg_category_path))
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup", type=int, default=0, help="warm up inference")
    parser.add_argument("--repeats", type=int, default=1, help="repeat number of inference")
    return parser.parse_args()

if __name__ == "__main__":
    main()
