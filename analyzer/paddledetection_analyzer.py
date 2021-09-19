import os
import sys
import subprocess
import re
import csv
import argparse
import logging

from paddle_frontend import paddle_frontend_supported_ops
from paddle_parser import get_ops

from collections import namedtuple
PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdparams operators unsupported_ops')

__dir__ = os.path.dirname(os.path.abspath(__file__))
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_config_file =os.path.abspath(os.path.join(__dir__, '../downloader/paddledet_filtered.csv'))
    default_static_models_path =os.path.abspath(os.path.join(__dir__, '../exporter/paddledet'))
    default_result_save_file =os.path.abspath(os.path.join(__dir__, './paddledet_filtered_operators.csv'))

    parser.add_argument("--config_file", type=str, default=default_config_file, help="The file we used to specify models need to analyse")
    parser.add_argument("--static_models_path", type=str, default=default_static_models_path, help="where to find the static models")
    parser.add_argument("--result_save_file", type=str, default=default_result_save_file, help="The file to save the analysis result")

    return parser.parse_args()

def convert_params_to_abspath(args):
    cwd = os.getcwd()
    if not os.path.isabs(args.config_file):
        args.config_file = os.path.abspath(os.path.join(cwd, args.config_file))
    if not os.path.isabs(args.static_models_path):
        args.static_models_path = os.path.abspath(os.path.join(cwd, args.static_models_path))
    if not os.path.isabs(args.result_save_file):
        args.result_save_file = os.path.abspath(os.path.join(cwd, args.result_save_file))

def welcome_info(args):
    logging.info("paddle detection models analyzer begin.")
    logging.debug("args.config_file: {}".format(args.config_file))
    logging.debug("args.static_models_path: {}".format(args.static_models_path))
    logging.debug("args.result_save_file: {}".format(args.result_save_file))


def main():
    #logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()
    convert_params_to_abspath(args)
    welcome_info(args)

    models = []
    with open(args.config_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            pdparams_url = row[2]
            pdparams_url = ''.join(pdparams_url.split()) # remove all whitespace
            if pdparams_url == 'None':
                models.append(PDModelInfo(row[0], row[1], row[2], 'ERROR', 'ERROR'))
                continue
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]

            exported_path = os.path.abspath(args.static_models_path + '/{}'.format(config_base))
            logging.info(exported_path)

            if os.path.exists(exported_path):
                operator_set = get_ops(exported_path)
                if operator_set is 'ERROR':
                    models.append(PDModelInfo(row[0], row[1], row[2], 'ERROR', 'ERROR'))
                    continue
                logging.debug(config_base, operator_set, len(operator_set))
                # pick out unsupported operators
                unsupported_ops = []
                for op in operator_set:
                    if op not in paddle_frontend_supported_ops:
                        unsupported_ops.append(op)

                models.append(PDModelInfo(row[0], row[1], row[2], ','.join(operator_set), ','.join(unsupported_ops) if len(unsupported_ops)>0 else 'None'))
            else: # keep ERROR
                models.append(PDModelInfo(row[0], row[1], row[2], 'ERROR', 'ERROR'))

    with open(args.result_save_file, 'w', newline='') as csvfile:
        # TODO: add title for each column
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(models)

if __name__ == '__main__':
    main()
    logging.info("paddle detection models analyzer end.")
