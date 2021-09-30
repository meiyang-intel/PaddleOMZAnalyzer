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

    print(exported_path, operator_set, len(operator_set), unsupported_ops, len(unsupported_ops))

    return sorted(operator_set), sorted(unsupported_ops)