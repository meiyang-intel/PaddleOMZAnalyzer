import os
import sys
import subprocess
import re
import csv
import gzip

# reference:
# https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md 
#
# python export.py \
#        --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
#        --model_path bisenet/model.pdparams
#        --save_dir output


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    print(__dir__)
    omz_dir = os.path.abspath(os.path.join(__dir__, '../../PaddleSeg'))
    sys.path.append(omz_dir)  # OMZ PaddleClas
    os.chdir(omz_dir)

    pipes = []

    with open(os.path.abspath(os.path.join(__dir__, '../downloader/paddleseg_full.csv')), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]

            #pdparams_url = row[2]
            pattern = re.compile(r"http.*.zip")
            if pattern.match(row[2]):
                zip_file_name = os.path.basename(row[2])
                dir_name = zip_file_name = os.path.splitext(zip_file_name)[0]
                zip_file_name = zip_file_name.replace('_with_softmax','')
                output_path = os.path.abspath(os.path.join(__dir__, 'paddleseg/'))
                zip_file_name = os.path.abspath(os.path.join(__dir__, '../downloader/paddleseg/{}.zip'.format(zip_file_name)))
                rm_command = "rm -rf " + output_path + "/" + config_base
                print(rm_command)
                os.system(rm_command)
                uzip_command = "unzip -d " + output_path + " " + zip_file_name + " && " + "mv "  + output_path + "/" + dir_name + " " + output_path + "/" + config_base
                print(uzip_command)
                os.system(uzip_command)
                continue

            pdparams_url = os.path.abspath(os.path.join(__dir__, '../downloader/paddleseg/{}.pdparams'.format(config_base)))
            output_path = os.path.abspath(os.path.join(__dir__, 'paddleseg/{}'.format(config_base)))

            exporter_cmd = 'python3 export.py --config {} --model_path {} --save_dir {}'.format(
                config_yaml, pdparams_url, output_path)
            print(exporter_cmd)
            p = subprocess.Popen(exporter_cmd, shell=True)
            pipes.append(p)

    exit_codes = [p.wait() for p in pipes]
    print(exit_codes) #TODO: check failure
