import os
import sys
import logging
import csv
import pathlib
import yaml
import argparse
from collections import namedtuple
import psutil
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
import utils

ModelInfo=namedtuple('ModelInfo', 'dlb_name pd_name path')

class DLBenchmark(object):
    def __init__(self, csv_files:list, model_dirs:list, dlb_path:str, dataset_dir:str, icv_dir:str, icv_bin_dir:str, output_dir:str):
        self.csv_files = csv_files
        self.model_dirs = model_dirs
        self.output_dir = output_dir
        self.dlb_path = dlb_path
        self.icv_bin_dir = icv_bin_dir
        self.icv_dir = icv_dir
        self.dataset_dir = dataset_dir

    def get_model_info(self):
        """
        Collect information of models in csv_files and dump them to self.model_info
        """
        self.model_info = []
        for csv_file in self.csv_files:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f, delimiter=',', quotechar='|')
                for row in reader:
                    dlb_name = row[0]
                    config_yaml = row[1]
                    config_yaml = ''.join(config_yaml.split()) # remove all whitespace
                    config_base = os.path.basename(config_yaml)
                    pd_name = os.path.splitext(config_base)[0]
                    for path in self.model_dirs:
                        p = str(list(pathlib.Path(path).glob('**/' + pd_name + '/*.pdmodel'))[0])
                        if p:
                            model_path = p
                    self.model_info.append(ModelInfo(dlb_name, pd_name, model_path))
        logging.info("Get model information done")

    def generate_topologies(self):
        """
        Generate topologies.yml according to self.model_info and information from DLB topologies.yml
        """
        # Retrieve model information of self.model_info from DLB topologies.yml. Save it to models_dict[]
        dlb_topologies = os.path.abspath(os.path.join(self.dlb_path, 'topologies.yml'))
        dlb_models = []
        with open(dlb_topologies, "r") as yaml_file:
            dlb_models = yaml.load(yaml_file, Loader=yaml.Loader)['topologies']

        models_dict = []
        for model_info in self.model_info:
            model_path = model_info.path
            weight_path = os.path.splitext(model_info.path)[0] + '.pdiparams'
            for model_i in range(len(dlb_models)):
                dlb_model = dlb_models[model_i]
                if dlb_model['name'] == model_info.dlb_name:
                    if 'models' in dlb_model:
                        for f_i in range(len(dlb_model['models'])):
                            framework = dlb_model['models'][f_i]
                            if framework['framework'] == 'paddle':
                                input_yml = os.path.join(self.dlb_path, framework['dataset'])
                        del dlb_model['models']
                    else:
                        input_yml = os.path.join(self.dlb_path, dlb_model['dataset'])

                    dlb_model['dataset'] = input_yml
                    dlb_model['model'] = model_path
                    dlb_model['weights'] = weight_path
                    dlb_model['framework'] = 'paddle'
                    dlb_model['readiness'] = 'Public'
                    models_dict.append(dlb_model)
        models_dict = {'topologies': models_dict}

        # dump models_dict[] to yaml file
        self.topologies_path = os.path.abspath(os.path.join(self.output_dir, 'topologies.yml'))
        with open(self.topologies_path, "w") as yaml_file:
            dump = yaml.dump(models_dict, default_flow_style = False, encoding = None)
            yaml_file.write(dump)
        logging.info("Generate topologies.yml")

    def run_dlb(self):
        """
        Run DLBenchmark
        """
        core_number = psutil.cpu_count(logical=False)
        dlb_cmd = 'python3 {}/scripts/run.py accuracy_check -m {} -d {} -c {} --icv_bin_dir {} \
                --launchers paddle_ov_cpu32 --runs_per_net 20 --time_per_net 0 \
                --timeout 1000 --batch_size 1 --batch_method reshape --nthreads {} \
                --frameworks paddle --tmpdir {} -o {}'.format(
                self.icv_dir, self.output_dir, self.dataset_dir, self.topologies_path, 
                self.icv_bin_dir, core_number, self.output_dir, self.output_dir)
        logging.info(dlb_cmd)
        try:
            p = subprocess.Popen(dlb_cmd, shell=True)
            p.wait()
            logging.info("Run dlbenchmark done.")
        except Exception as e:
            logging.error("Run dlbenchmark failed: " + str(e))

    def run(self):
        self.get_model_info()
        self.generate_topologies()
        self.run_dlb()

def list_type(vstr, sep=','):
    """
    Return a list seperated with 'sep'
    """
    return vstr.split(sep)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--csv_files", type=list_type, help="List of csv files containing models information need to run")
    parser.add_argument("--model_dirs", type=list_type, help="List of model path which contains the models")
    parser.add_argument("--dlb_path", type=str, help="Path of DLBenchmark models")
    parser.add_argument("--dataset_dir", type=str, help="Path of dataset")
    parser.add_argument("--icv_dir", type=str, help="Path of DLBenchmark tools")
    parser.add_argument("--icv_bin_dir", type=str, help="Path containing DLBenchmark binary")
    parser.add_argument("--output_dir", type=str, help="Output path")

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()
    dlb = DLBenchmark(args.csv_files, args.model_dirs, args.dlb_path, args.dataset_dir, 
                    args.icv_dir, args.icv_bin_dir, args.output_dir)
    dlb.run()
