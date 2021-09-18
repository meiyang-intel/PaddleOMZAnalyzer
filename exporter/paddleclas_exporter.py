import os
import sys
import subprocess
import re
import csv
import argparse
import logging

# reference:
# https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict
#
# python tools/export_model.py \
#     --model MobileNetV3_large_x1_0 \
#     --pretrained_model ./output/MobileNetV3_large_x1_0/best_model/ppcls \
#     --output_path ./inference \
#     --class_dim 1000

__dir__ = os.path.dirname(os.path.abspath(__file__))
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_project_dir =os.path.abspath(os.path.join(__dir__, '../../PaddleClas'))
    default_config_file =os.path.abspath(os.path.join(__dir__, '../downloader/paddleclas_filtered.csv'))
    default_dynamic_models_path =os.path.abspath(os.path.join(__dir__, '../downloader/paddleclas'))
    default_static_models_save_path =os.path.abspath(os.path.join(__dir__, './paddleclas'))

    parser.add_argument("--project_dir", type=str, default=default_project_dir, help="PaddleClas project directory")
    parser.add_argument("--config_file", type=str, default=default_config_file, help="The file we used to specify models need to export")
    parser.add_argument("--dynamic_models_path", type=str, default=default_dynamic_models_path, help="where to find the dynamic models")
    parser.add_argument("--static_models_save_path", type=str, default=default_static_models_save_path, help="where to save the static models we exported")
    parser.add_argument("--subprocess_max_number", type=int, default=5, help="subprocess max number which used to limit the process number to prevent it consumes all cpu/memory resources")

    return parser.parse_args()

def convert_params_to_abspath(args):
    cwd = os.getcwd()
    if not os.path.isabs(args.project_dir):
        args.project_dir = os.path.abspath(os.path.join(cwd, args.project_dir))
    if not os.path.isabs(args.config_file):
        args.config_file = os.path.abspath(os.path.join(cwd, args.config_file))
    if not os.path.isabs(args.dynamic_models_path):
        args.dynamic_models_path = os.path.abspath(os.path.join(cwd, args.dynamic_models_path))
    if not os.path.isabs(args.static_models_save_path):
        args.static_models_save_path = os.path.abspath(os.path.join(cwd, args.static_models_save_path))

def welcome_info(args):
    logging.info("paddle classification models exporter begin.")
    logging.debug("args.project_dir: {}".format(args.project_dir))
    logging.debug("args.config_file: {}".format(args.config_file))
    logging.debug("args.dynamic_models_path: {}".format(args.dynamic_models_path))
    logging.debug("args.static_models_save_path: {}".format(args.static_models_save_path))
    logging.debug("args.subprocess_max_number: {}".format(args.subprocess_max_number))


def main():
    #logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()
    convert_params_to_abspath(args)
    welcome_info(args)

    sys.path.append(args.project_dir)  # OMZ PaddleClas
    os.chdir(args.project_dir)

    pipes = []
    sum_exit_codes = []
    with open(args.config_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            pdparams_url = row[2]
            pdparams_url = ''.join(pdparams_url.split()) # remove all whitespace
            if pdparams_url == 'None':
                continue
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]
            file_name = config_base + '_pretrained'
            pretraind_model = os.path.abspath(args.dynamic_models_path + '/{}'.format(file_name))
            output_path = os.path.abspath(args.static_models_save_path + '/{}'.format(config_base))

            exporter_cmd = 'python3 tools/export_model.py --model {} --pretrained_model {} --output_path {} --class_dim 1000'.format(
                config_base, pretraind_model, output_path)
            logging.info('exporter_cmd: {}'.format(exporter_cmd))
            p = subprocess.Popen(exporter_cmd, shell=True)
            pipes.append(p)
            if len(pipes) == args.subprocess_max_number:
                exit_codes = [p.wait() for p in pipes]
                sum_exit_codes = sum_exit_codes + exit_codes
                pipes.clear()

        if len(pipes) != 0:
            exit_codes = [p.wait() for p in pipes]
            sum_exit_codes = sum_exit_codes + exit_codes

    logging.info('exit_codes: {}'.format(sum_exit_codes))

if __name__ == '__main__':
    main()
    logging.info("paddle classification models exporter end.")

