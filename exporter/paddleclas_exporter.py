import os
import sys
import subprocess
import re
import csv

# reference:
# https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict
#
# python tools/export_model.py \
#     --model MobileNetV3_large_x1_0 \
#     --pretrained_model ./output/MobileNetV3_large_x1_0/best_model/ppcls \
#     --output_path ./inference \
#     --class_dim 1000


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    omz_dir = os.path.abspath(os.path.join(__dir__, '../../PaddleClas'))
    sys.path.append(omz_dir)  # OMZ PaddleClas
    os.chdir(omz_dir)

    pipes = []

    with open(os.path.abspath(os.path.join(__dir__, '../downloader/paddleclas_full.csv')), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]

            pdparams_url = row[2]
            pdprams_name = os.path.basename(pdparams_url)
            pdprams_name = os.path.splitext(pdprams_name)[0]            
            pretraind_model = os.path.abspath(os.path.join(__dir__, '../downloader/paddleclas/{}'.format(pdprams_name)))

            output_path = os.path.abspath(os.path.join(__dir__, 'paddleclas/{}'.format(config_base)))

            exporter_cmd = 'python3 tools/export_model.py --model {} --pretrained_model {} --output_path {} --class_dim 1000'.format(
                config_base, pretraind_model, output_path)
            print(exporter_cmd)
            p = subprocess.Popen(exporter_cmd, shell=True)
            pipes.append(p)

    exit_codes = [p.wait() for p in pipes]
    print(exit_codes) #TODO: check failure
