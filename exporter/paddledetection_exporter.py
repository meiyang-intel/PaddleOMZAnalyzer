import os
import sys
import subprocess
import re
import csv
import pathlib
import argparse
import logging

# reference:
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md
#
# python tools/export_model.py \
# -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml \
# -o use_gpu=false weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams \
# TestReader.inputs_def.image_shape=[3,608,608] \
# --output_dir ../export_pdpd/

# reference:
# https://github.com/PaddlePaddle/PaddleDetection/blob/f0a30f3ba6095ebfdc8fffb6d02766406afc438a/docs/tutorials/PrepareDataSet.md#COCO%E6%95%B0%E6%8D%AE
# to download coco dataset, which is required to export configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_640.yml.

__dir__ = os.path.dirname(os.path.abspath(__file__))
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_project_dir =os.path.abspath(os.path.join(__dir__, '../../PaddleDetection'))
    default_config_file =os.path.abspath(os.path.join(__dir__, '../downloader/paddledet_filtered.csv'))
    default_dynamic_models_path =os.path.abspath(os.path.join(__dir__, '../downloader/paddledet'))
    default_static_models_save_path =os.path.abspath(os.path.join(__dir__, './paddledet'))

    parser.add_argument("--project_dir", type=str, default=default_project_dir, help="PaddleDetection project directory")
    parser.add_argument("--config_file", type=str, default=default_config_file, help="The file we used to specify models need to export")
    parser.add_argument("--dynamic_models_path", type=str, default=default_dynamic_models_path, help="where to find the dynamic models")
    parser.add_argument("--static_models_save_path", type=str, default=default_static_models_save_path, help="where to save the static models we exported")
    parser.add_argument("--invalidate_cache", type=bool, default=False, help="if ignore the cache or not, program will reexport the dynamic models even it have been exported before when you set it True value.")
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
    logging.info("paddle detection models exporter begin.")
    logging.debug("args.project_dir: {}".format(args.project_dir))
    logging.debug("args.config_file: {}".format(args.config_file))
    logging.debug("args.dynamic_models_path: {}".format(args.dynamic_models_path))
    logging.debug("args.static_models_save_path: {}".format(args.static_models_save_path))
    logging.debug("args.invalidate_cache: {}".format(args.invalidate_cache))
    logging.debug("args.subprocess_max_number: {}".format(args.subprocess_max_number))


def main():
    #logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()
    convert_params_to_abspath(args)
    welcome_info(args)

    sys.path.append(args.project_dir)  # OMZ PaddleDetection
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

            # 'static/configs' are handled in its relative path
            p = pathlib.Path(config_yaml)
            work_dir = os.path.realpath(os.path.join(args.project_dir, 'static')) if p.parts[0]=='static' else args.project_dir
            if p.parts[0]=='static': config_yaml = str(pathlib.Path(*p.parts[1:]))  # remove 'static' from relative path

            if args.invalidate_cache is False:
                config_base = os.path.basename(config_yaml)
                config_base = os.path.splitext(config_base)[0]
                model_path = os.path.join(args.static_models_save_path, config_base)
                if os.path.exists(model_path) and (os.path.exists(os.path.join(model_path, 'model.pdmodel')) or os.path.exists(os.path.join(model_path, '__model__'))): # cached
                    logging.info('INFO: {} already exported. Ingnore it this time.'.format(config_base))
                    continue

            # TODO: it might consumes all cpu/memory resources, so we'd better limit the size of concurrent working pipes.
            exporter_cmd = 'chdir {} && python3 tools/export_model.py -c {} -o use_gpu=false weights={} TestReader.inputs_def.image_shape=[3,608,608] --output_dir {}'.format(
                work_dir, config_yaml, pdparams_url, args.static_models_save_path)
            logging.info(exporter_cmd)
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
    logging.info("paddle detection models exporter end.")
