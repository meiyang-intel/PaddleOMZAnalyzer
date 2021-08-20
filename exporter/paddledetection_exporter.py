import os
import sys
import subprocess
import re
import csv

# reference:
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md
# 
# python tools/export_model.py \
# -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml \
# -o use_gpu=false weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams \
# TestReader.inputs_def.image_shape=[3,608,608] \
# --output_dir ../export_pdpd/

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    omz_dir = os.path.abspath(os.path.join(__dir__, '../../PaddleDetection'))
    sys.path.append(omz_dir)  # OMZ PaddleDetection
    os.chdir(omz_dir)

    pipes = []

    with open(os.path.abspath(os.path.join(__dir__, '../downloader/paddledet_filtered.csv')), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]

            pdparams_url = row[2]
            # pdprams_name = os.path.basename(pdparams_url)
            # pdprams_name = os.path.splitext(pdprams_name)[0]            
            # pretraind_model = os.path.abspath(os.path.join(__dir__, '../downloader/paddledet/{}'.format(pdprams_name)))

            output_path = os.path.abspath(os.path.join(__dir__, 'paddledet')) # paddledetection will append modelname as its exported dirname.

            exporter_cmd = 'python3 tools/export_model.py -c {} -o use_gpu=false weights={} TestReader.inputs_def.image_shape=[3,608,608] --output_dir {}'.format(
                config_yaml, pdparams_url, output_path)
            print(exporter_cmd)
            p = subprocess.Popen(exporter_cmd, shell=True)
            pipes.append(p)

    exit_codes = [p.wait() for p in pipes]
    print(exit_codes) #TODO: check failure
