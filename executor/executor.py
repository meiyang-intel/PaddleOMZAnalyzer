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
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertTokenizer
# from paddle.inference import Config
# from paddle.inference import create_predictor

from abc import ABCMeta, abstractmethod

from utils import is_float_tensor, randtool, get_tensor_dtype
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork, StatusCode
from collections import namedtuple

class Executor(object):
    __metaclass__ = ABCMeta

    def __init__(self, pdmodel:str):
        self.__pdmodel__ = pdmodel

        self.inputs_info = dict()  # shape, dtype of inputs
        self.outputs_info = dict() # shape, dtype of outputs
        self.inference_results = dict() # note: make inference results of each executor be dict for comparision.
        self.warmup_time = None
        self.repeat_time = None

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

# helper
def query_inputs_info(pdmodel):
    paddle.enable_static()

    # load model
    exe = paddle.static.Executor(paddle.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(
                        os.path.splitext(pdmodel)[0], # pdmodel prefix
                        exe)

    print(feed_target_names, fetch_targets)

    # 输出计算图所有input结点信息
    inputs_info = dict()
    for ipt in feed_target_names:
        var=inference_program.global_block().var(ipt)
        print('model input name {} with shape {}, dtype {}'.format(ipt, var.shape, var.dtype))
        inputs_info[ipt] = (var.shape, var.dtype)

    return inputs_info, exe, inference_program, fetch_targets

"""
obsolete and replaced by PaddlePredictorExecutor
"""
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
        self.inputs_info, self.exe, self.inference_program, self.fetch_targets = query_inputs_info(pdmodel)

        # workaround for yolo and ppyolo,
        # which requires return_numpy False, else exception prompts from Paddle.
        self.return_numpy = True
        # 输出计算图所有结点信息
        for i, op in enumerate(self.inference_program.blocks[0].ops):
            if op.type in { 'matrix_nms', 'multiclass_nms3', 'multiclass_nms2', 'multiclass_nms' }:
                self.return_numpy = False

    def inference(self, inputs:dict, warmup=0, repeats=1):
        t0 = time.time()
        #warm up
        for i in range(warmup):
            inference_results = self.exe.run(self.inference_program, feed=inputs, fetch_list=self.fetch_targets, return_numpy = self.return_numpy)

        t1 = time.time()
        if warmup:
            self.warmup_time = (t1 - t0) * 1000.0 / warmup

        #repeats
        for i in range(repeats):
            inference_results = self.exe.run(self.inference_program, feed=inputs, fetch_list=self.fetch_targets, return_numpy = self.return_numpy)
        t2 = time.time()
        ms = (t2 - t1) * 1000.0 / repeats
        #print("PaddleExecutor Inference: {} ms per batch image".format(ms))

        # debug info
        #print("PaddleExecutor inference results type {}, len {}, type of element {}".format(type(inference_results), len(inference_results), type(inference_results[0])))

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
        config = paddle_infer.Config(self.__pdmodel__, self.pdiparams)

        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)
        # cache 10 different shapes for mkldnn to avoid memory leak
        config.set_mkldnn_cache_capacity(10)
        config.enable_mkldnn()

        # # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)

        self.predictor = paddle_infer.create_predictor(config)

        # Paddle-private
        self.inputs_info, _, _, _ = query_inputs_info(pdmodel)

    def inference(self, inputs:dict, warmup=0, repeats=1):
        input_names = self.predictor.get_input_names()
        for input_name in input_names:
            input_handle = self.predictor.get_input_handle(input_name)
            input_handle.reshape(inputs[input_name].shape)
            input_handle.copy_from_cpu(inputs[input_name])

        t0 = time.time()
        #warmup
        for i in range(warmup):
            self.predictor.run()

        t1 = time.time()
        if warmup:
            self.warmup_time = (t1 - t0) * 1000.0 / warmup

        #repeats
        for i in range(repeats):
            self.predictor.run()
        t2 = time.time()
        self.repeat_time = (t2 - t1) * 1000.0 / repeats
        #print("PaddlePredictorExecutor Inference: {} ms per batch image".format(self.repeat_time))

        output_names = self.predictor.get_output_names()
        for output_name in output_names:
            output_handle = self.predictor.get_output_handle(output_name)
            self.inference_results[output_name] = output_handle.copy_to_cpu()

class PaddlenlpPredictor(PaddlePredictorExecutor):
    def convert_example(self, example, tokenizer, max_seq_length=128):
        text = example
        encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        segment_ids = encoded_inputs["token_type_ids"]

        return input_ids, segment_ids

    def generate_inputs(self, batch_size, language: str = 'English'):
        test_inputs = dict()

        if language == 'Chinese':
            data = [
                    '今天早上天气很好',
                    '昨天真的好冷',
                    '今天下午风就开始了，很大很大的风',
                    '一觉睡到大天亮的话，精神一定会很饱满',
                    '红豆生南国，春来发几枝，两支。'
                    ]
        elif language == 'English':
            data = [
                    'against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting',
                    'the situation in a well-balanced fashion',
                    'at achieving the modest , crowd-pleasing goals it sets for itself',
                    'so pat it makes your teeth hurt',
                    'this new jangle of noise , mayhem and stupidity must be a serious contender for the title .'
                    ]
        else:
            print("No support \"{}\" language".format(language))

        label_map = {0: 'negative', 1: 'positive'}

        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.dirname(self.__pdmodel__))

        examples = []
        for text in data:
            input_ids, segment_ids = self.convert_example(
                text,
                self.tokenizer)
            examples.append((input_ids, segment_ids))

        # Seperates data into some batches.
        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]

        # create lambda
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),  # segment
        ): fn(samples)

        # create input info
        input_ids, segment_ids = batchify_fn(batches[0])
        test_inputs["input_ids"] = input_ids
        test_inputs["token_type_ids"] = segment_ids

        return test_inputs

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
        assert isinstance(self.exec_net, ExecutableNetwork)

        t0 = time.time()
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

        t1 = time.time()
        if warmup:
            self.warmup_time = (t1 - t0) * 1000.0 / warmup

        #inference
        infer_requests = self.exec_net.requests
        iteration = 0

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
        self.repeat_time = (t2 - t1) * 1000.0 / repeats
        #print("OpenvinoExecutor Inference: {} ms per batch image".format(self.repeat_time))

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
            print('No Equal.')
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