import os
import sys
import subprocess
import re
import csv

from paddle_frontend import paddle_frontend_supported_ops
from paddle_parser import get_ops

from collections import namedtuple
PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdparams operators unsupported_ops')

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))

    # print(sorted(paddle_frontend_supported_ops),len(paddle_frontend_supported_ops))
    models = []
    with open(os.path.abspath(os.path.join(__dir__, '../downloader/paddledet_filtered.csv')), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]

            exported_path = os.path.abspath(os.path.join(__dir__, '../exporter/paddledet/{}'.format(config_base)))
            print(exported_path)

            if os.path.exists(exported_path):
                operator_set = get_ops(exported_path)
                print(config_base, operator_set, len(operator_set))

                # pick out unsupported operators
                unsupported_ops = []
                for op in operator_set:
                    if op not in paddle_frontend_supported_ops:
                        unsupported_ops.append(op)

                models.append(PDModelInfo(row[0], row[1], row[2], ','.join(operator_set), ','.join(unsupported_ops) if len(unsupported_ops)>0 else 'None'))
            else: # keep ERROR
                models.append(PDModelInfo(row[0], row[1], row[2], 'ERROR', 'ERROR'))

    with open('paddledet_operators.csv', 'w', newline='') as csvfile:
        # TODO: add title for each column
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(models)