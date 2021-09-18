import os
import sys
import subprocess
import re
import csv
import argparse
import logging

# reference:
# https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md
#
# python export.py \
#        --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
#        --model_path bisenet/model.pdparams
#        --save_dir output

__dir__ = os.path.dirname(os.path.abspath(__file__))
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_project_dir =os.path.abspath(os.path.join(__dir__, '../../PaddleSeg'))
    default_config_file =os.path.abspath(os.path.join(__dir__, '../downloader/paddleseg_filtered.csv'))
    default_dynamic_models_path =os.path.abspath(os.path.join(__dir__, '../downloader/paddleseg'))
    default_static_models_save_path =os.path.abspath(os.path.join(__dir__, './paddleseg'))

    parser.add_argument("--project_dir", type=str, default=default_project_dir, help="PaddleSeg project directory")
    parser.add_argument("--config_file", type=str, default=default_config_file, help="The file we used to specify models need to export")
    parser.add_argument("--dynamic_models_path", type=str, default=default_dynamic_models_path, help="where to find the dynamic models")
    parser.add_argument("--static_models_save_path", type=str, default=default_static_models_save_path, help="where to save the static models we exported")

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
    logging.info("paddle segmentation models exporter begin.")
    logging.debug("args.project_dir: {}".format(args.project_dir))
    logging.debug("args.config_file: {}".format(args.config_file))
    logging.debug("args.dynamic_models_path: {}".format(args.dynamic_models_path))
    logging.debug("args.static_models_save_path: {}".format(args.static_models_save_path))


def main():
    #logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()
    convert_params_to_abspath(args)
    welcome_info(args)

    sys.path.append(args.project_dir)
    os.chdir(args.project_dir)

    pipes = []

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

            #pdparams_url = row[2]
            pattern = re.compile(r"http.*.zip")
            if pattern.match(pdparams_url):
                dir_name = os.path.basename(pdparams_url)
                dir_name = os.path.splitext(dir_name)[0]
                output_path = os.path.abspath(os.path.join(__dir__, 'paddleseg/'))
                zip_file_name = os.path.abspath(os.path.join(__dir__, '../downloader/paddleseg/{}.zip'.format(config_base)))
                rm_command = "rm -rf " + args.static_models_save_path + "/" + config_base
                logging.info(rm_command)
                os.system(rm_command)
                uzip_command = "unzip -d " + args.static_models_save_path + " " + zip_file_name + " && " + "mv "  + args.static_models_save_path + "/" + dir_name + " " + args.static_models_save_path + "/" + config_base
                logging.info(uzip_command)
                os.system(uzip_command)
                continue
            pdparams_url = os.path.abspath(args.dynamic_models_path + '/{}.pdparams'.format(config_base))
            output_path = os.path.abspath(args.static_models_save_path + '/{}'.format(config_base))

            exporter_cmd = 'python3 export.py --config {} --model_path {} --save_dir {}'.format(
                config_yaml, pdparams_url, output_path)
            logging.info(exporter_cmd)
            p = subprocess.Popen(exporter_cmd, shell=True)
            pipes.append(p)

    exit_codes = [p.wait() for p in pipes]
    logging.info(exit_codes) #TODO: check failure

if __name__ == '__main__':
    main()
    logging.info("paddle segmentation models exporter end.")
