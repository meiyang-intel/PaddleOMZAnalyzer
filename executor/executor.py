import argparse
import os
import sys
import logging
import numpy as np
from pathlib import Path
import csv

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import paddle
# from paddle.inference import Config
# from paddle.inference import create_predictor

from abc import ABCMeta, abstractmethod

from utils import is_float_tensor, randtool, get_tensor_dtype
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork
from collections import namedtuple
#PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdparams compare_result')
PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdmodel compare_result')

class Executor(object):
    __metaclass__ = ABCMeta

    def __init__(self, pdmodel:str):
        self.__pdmodel__ = pdmodel

        self.inputs_info = dict()  # shape, dtype of inputs
        self.outputs_info = dict() # shape, dtype of outputs
        self.inference_results = None # inference results, may vary from excutor inheriates.??

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
    def inference(self, inputs:dict, warmup=0, benchmarking=False):
        """
        Inferencing
        inputs [in] a dict of input name and tensor.
        """
        pass

    def run(self, test_inputs):
        """
        Generate input randomly, and inferencing
        """

        # inference
        self.inference(test_inputs)

class PaddleExecutor(Executor):
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

    def inference(self, inputs:dict, warmup=0, benchmarking=False):
        # run
        self.inference_results = self.exe.run(self.inference_program, feed=inputs, fetch_list=self.fetch_targets, return_numpy=False)

class OpenvinoExecutor(Executor):
    def __init__(self, pdmodel):
        super().__init__(pdmodel)
        self.ie = IECore()
        self.net = self.ie.read_network(self.__pdmodel__)

    def inference(self, inputs:dict, warmup=0, benchmarking=False):
        reshape_dict = {}
        for item in list(self.net.input_info.items()):
            reshape_dict[item[0]]=inputs[item[0]].shape
        self.net.reshape(reshape_dict)
        self.exec_net = self.ie.load_network(self.net, 'CPU') # device
        assert isinstance(self.exec_net, ExecutableNetwork)
        self.inference_results = self.exec_net.infer(inputs)
        self.inference_results = list(self.inference_results.values())

def compare(result, expect, delta=1e-6, rtol=1e-6):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return: True or False
    """
    res = True
    if type(result) == np.ndarray:
        expect = np.array(expect)
        strDtype = str(result.dtype)
        if strDtype.startswith('int') or strDtype.startswith('uint'):
            rtol = 0
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=False)
        return res
    elif type(result) == list:
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                res = compare(result[i], expect[i], delta, rtol)
            else:
                res = compare(result[i].numpy(), expect[i], delta, rtol)

            if res is not True:
                return False
    else:
        print("Compare Format Error.")
        return False

    return True

def inference_and_compare(model_file, batch_size):
    if os.path.exists(model_file):
        #inference
        ## paddle inference
        pdpd_executor = PaddleExecutor(model_file)
        test_inputs = pdpd_executor.generate_inputs(batch_size)
        pdpd_executor.run(test_inputs)
        pdpd_result = pdpd_executor.get_inference_results()

        ## paddle inference
        ov_executor = OpenvinoExecutor(model_file)
        ov_executor.run(test_inputs)
        ov_result = ov_executor.get_inference_results()

        # compare openvino result and paddle result
        return compare(ov_result, pdpd_result)
    else:
        return False

def loop_inference_and_compare(model_category: str, batch_size=1):
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

                res = inference_and_compare(models[0], batch_size)
                compare_result_list.append(PDModelInfo(row[0], row[1], models[0], "Equal" if res else 'No Equal'))
            else:
                print('Have no this {} directory'.format(exported_path))

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(compare_result_list)

def main():
    args = parse_args()
    if args.mode == 'all':
        loop_inference_and_compare('classify', args.batch_size)
        loop_inference_and_compare('detection', args.batch_size)
        loop_inference_and_compare('segmentation', args.batch_size)
    elif args.mode == 'single':
        res = inference_and_compare(args.model_file, args.batch_size)
        if res:
            print("Equal")
        else:
            print("Not Equal")
    elif args.mode == 'category':
        if args.category == 'classify':
            loop_inference_and_compare('classify', args.batch_size)
        elif args.category == 'detection':
            loop_inference_and_compare('detection', args.batch_size)
        elif args.category == 'segmentation':
            loop_inference_and_compare('segmentation', args.batch_size)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    det_category_path =os.path.abspath(os.path.join(__dir__, '../exporter/paddledet'))
    clas_category_path =os.path.abspath(os.path.join(__dir__, '../exporter/paddleclas'))
    seg_category_path =os.path.abspath(os.path.join(__dir__, '../exporter/paddleseg'))
    parser.add_argument("--mode", type=str, default='single', choices=['single', 'category', 'all'], help="single: only execute the single model_file you specify\ncategory: execute all models belong to the category you specify\nall: execute all models in all categorys\n(default:single)")
    parser.add_argument("--model_file", type=str, default=os.path.join(__dir__, '../exporter/paddleclas/MobileNetV1/inference.pdmodel'), help="model filename, only effect on single mode")
    parser.add_argument("--category", type=str, default='classify', choices=['classify', 'detection', 'segmentation'], help="classify: execute all models in {}\ndetection: execute all models in {}\nsegmentation: execute all models in {}\n(default:classify)\nonly effect on category mode".format(clas_category_path, det_category_path,  seg_category_path))
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    main()




