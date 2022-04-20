import csv
import os
import argparse
from glob import glob

from paddle_frontend import paddle_frontend_supported_ops
from paddle_parser import get_ops

# return all_ops and unsupported_ops in model.
def parse_model_ops(exported_path):
    operator_set = get_ops(exported_path)    

    # pick out unsupported operators
    unsupported_ops = []
    for op in operator_set:
        if op not in paddle_frontend_supported_ops:
            unsupported_ops.append(op)

    # print(exported_path, operator_set, len(operator_set), unsupported_ops, len(unsupported_ops))

    return sorted(operator_set), sorted(unsupported_ops)


__dir__ = os.path.dirname(os.path.abspath(__file__))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='', help="model dir")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if os.path.isdir(args.model_dir):
        operator_set, unsupported_ops = parse_model_ops(args.model_dir)

        print(operator_set, len(operator_set))
        print(unsupported_ops, len(unsupported_ops))


#usage
#
#python3 parser.py --model_dir=../exporter/paddleocr/ch_ppocr_mobile_v2.0_cls_infer/
