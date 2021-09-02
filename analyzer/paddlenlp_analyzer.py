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

    # transformer
    # base
    test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/transformer/base'))
    operator_set, unsupported_ops = parse_model_ops(test_model)
    # big
    test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/transformer/big'))
    operator_set, unsupported_ops = parse_model_ops(test_model)


'''
model operator_set len(operator_set) unsupported_ops len(unsupported_ops)
paddlenlp/bert ['cast', 'cumsum', 'dropout', 'elementwise_add', 'elementwise_sub', 'equal', 'fill_any_like', 'fill_constant', 'gelu', 'layer_norm', 'lookup_table_v2', 'matmul', 'matmul_v2', 'reshape2', 'scale', 'slice', 'softmax', 'tanh', 'transpose2', 'unsqueeze2'] 20 ['cumsum', 'fill_any_like', 'gelu', 'layer_norm', 'lookup_table_v2', 'matmul_v2', 'tanh'] 7
paddlenlp/waybill_ie ['cast', 'cumsum', 'dropout', 'elementwise_add', 'elementwise_sub', 'equal', 'fill_any_like', 'fill_constant', 'layer_norm', 'lookup_table_v2', 'matmul', 'matmul_v2', 'relu', 'reshape2', 'scale', 'softmax', 'transpose2', 'unsqueeze2'] 18 ['cumsum', 'fill_any_like', 'layer_norm', 'lookup_table_v2', 'matmul_v2'] 5
paddlenlp/transformer/base ['assign_value', 'cast', 'dropout', 'elementwise_add', 'elementwise_mul', 'equal', 'expand', 'fill_constant', 'fill_constant_batch_size_like', 'fill_zeros_like', 'gather_tree', 'layer_norm', 'logical_not', 'lookup_table_v2', 'matmul', 'matmul_v2', 'not_equal', 'range', 'reduce_all', 'relu', 'reshape2', 'scale', 'shape', 'slice', 'softmax', 'tensor_array_to_tensor', 'transpose2', 'unsqueeze2', 'while'] 29 ['expand', 'fill_zeros_like', 'gather_tree', 'layer_norm', 'lookup_table_v2', 'matmul_v2', 'not_equal', 'reduce_all', 'tensor_array_to_tensor', 'while'] 10
paddlenlp/transformer/big ['assign_value', 'cast', 'dropout', 'elementwise_add', 'elementwise_mul', 'equal', 'expand', 'fill_constant', 'fill_constant_batch_size_like', 'fill_zeros_like', 'gather_tree', 'layer_norm', 'logical_not', 'lookup_table_v2', 'matmul', 'matmul_v2', 'not_equal', 'range', 'reduce_all', 'relu', 'reshape2', 'scale', 'shape', 'slice', 'softmax', 'tensor_array_to_tensor', 'transpose2', 'unsqueeze2', 'while'] 29 ['expand', 'fill_zeros_like', 'gather_tree', 'layer_norm', 'lookup_table_v2', 'matmul_v2', 'not_equal', 'reduce_all', 'tensor_array_to_tensor', 'while'] 10
'''
