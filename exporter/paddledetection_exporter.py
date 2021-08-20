import os
import sys
import subprocess
import re
import csv
import pathlib

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

invalidate_cache = False # default

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

            pdparams_url = row[2]
            output_path = os.path.abspath(os.path.join(__dir__, 'paddledet')) # paddledetection will append modelname as its exported dirname.

            # 'static/configs' are handled in its relative path
            p = pathlib.Path(config_yaml)
            work_dir = os.path.realpath(os.path.join(omz_dir, 'static')) if p.parts[0]=='static' else omz_dir
            if p.parts[0]=='static': config_yaml = str(pathlib.Path(*p.parts[1:]))  # remove 'static' from relative path

            if invalidate_cache is False:
                config_base = os.path.basename(config_yaml)
                config_base = os.path.splitext(config_base)[0]    
                model_path = os.path.join(output_path, config_base)
                if os.path.exists(model_path) and (os.path.exists(os.path.join(model_path, 'model.pdmodel')) or os.path.exists(os.path.join(model_path, '__model__'))): # cached
                    print('INFO: {} already exported. Ingnore it this time.'.format(config_base))
                    continue
            
            # TODO: it might consumes all cpu/memory resources, so we'd better limit the size of concurrent working pipes.
            exporter_cmd = 'chdir {} && python3 tools/export_model.py -c {} -o use_gpu=false weights={} TestReader.inputs_def.image_shape=[3,608,608] --output_dir {}'.format(
                work_dir, config_yaml, pdparams_url, output_path)
            print(exporter_cmd)
            p = subprocess.Popen(exporter_cmd, shell=True)
            pipes.append(p)

    exit_codes = [p.wait() for p in pipes]
    print(exit_codes) #TODO: check failure
