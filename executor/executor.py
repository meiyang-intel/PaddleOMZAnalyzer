import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import paddle
# from paddle.inference import Config
# from paddle.inference import create_predictor

from abc import ABCMeta, abstractmethod

from utils import is_float_tensor, randtool, get_tensor_dtype

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

    def run(self, batch_size):
        """
        Generate input randomly, and inferencing
        """
        test_inputs = self.generate_inputs(batch_size)

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
        self.inference_results = self.exe.run(self.inference_program, feed=inputs, fetch_list=self.fetch_targets)
       
def main():
    args = parse_args()

    # model_path
    test_model = os.path.abspath(args.model_file)

    pdpd_executor = PaddleExecutor(test_model)
    pdpd_executor.run(args.batch_size)
    pdpd_result = pdpd_executor.get_inference_results()
    print('\nresult of paddle:', type(pdpd_result))
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, default=os.path.join(__dir__, '../exporter/paddleclas/MobileNetV1/inference.pdmodel'), help="model filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args() 

if __name__ == "__main__":
    main()      