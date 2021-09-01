import os
import sys
import subprocess
import re
import csv

from paddle_frontend import paddle_frontend_supported_ops
from paddle_parser import get_ops

from collections import namedtuple
PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdparams operators unsupported_ops')

# return all_ops and unsupported_ops in model.
def parse_model_ops(exported_path):
    operator_set = get_ops(exported_path)    

    # pick out unsupported operators
    unsupported_ops = []
    for op in operator_set:
        if op not in paddle_frontend_supported_ops:
            unsupported_ops.append(op)

    print(exported_path, operator_set, len(operator_set), unsupported_ops, len(unsupported_ops))

    return operator_set, unsupported_ops

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))

    #BERT
    bert_all = ['bert-base-uncased', 
                'bert-large-uncased', 
                'bert-base-cased',
                'bert-large-cased',
                'bert-base-multilingual-uncased',
                'bert-base-multilingual-cased',
                'bert-base-chinese',
                'bert-wwm-chinese',
                'bert-wwm-ext-chinese',
                'simbert-base-chinese'
                ]
    for bert in bert_all:
        test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/{}'.format(bert)))
        if os.path.exists(test_model):
            operator_set, unsupported_ops = parse_model_ops(test_model)

    # waybill_ie
    test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/waybill_ie'))
    operator_set, unsupported_ops = parse_model_ops(test_model)
