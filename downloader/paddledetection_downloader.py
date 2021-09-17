import argparse
import os
import sys
import re
import logging
import csv
from pathlib import Path
from bs4 import BeautifulSoup
import markdown
import requests

from downloader_base import PDFiltedModelInfo, PDAllModelInfo, base_downloader

__dir__ = os.path.dirname(os.path.abspath(__file__))

class paddledet_downloader(base_downloader):
    def __init__(self, project_dir, filter_data_file, download_mode, models_download_path, full_model_info_save_file, filtered_model_info_save_file):
        super().__init__(project_dir, filter_data_file, download_mode, models_download_path, full_model_info_save_file, filtered_model_info_save_file)
        self.sum_yml_and_yaml_list = []

    def merge_yml_list_and_yaml_list(self):
        """
        merge the self.all_yaml_file_list and the self.all_yml_file_list
        to one self.sum_yml_and_yaml_list
        """
        if len(self.sum_yml_and_yaml_list) > 0:
            return self.sum_yml_and_yaml_list

        for yaml_file in self.all_yaml_file_list:
            self.sum_yml_and_yaml_list.append(yaml_file)

        for temp_file_name in self.all_yml_file_list:
            self.sum_yml_and_yaml_list.append(temp_file_name)

        return self.sum_yml_and_yaml_list

    def get_all_model_info_list(self):
        """
        match self.all_yml_file_list and self.all_yaml_file_list with self.all_pdparams_urls,
        then get the self.all_model_info_list
        """
        if len(self.all_model_info_list) > 0:
            return self.all_model_info_list

        self.merge_yml_list_and_yaml_list()

        for temp_file_name in self.sum_yml_and_yaml_list:
            config_base = os.path.basename(temp_file_name)
            config_base = os.path.splitext(config_base)[0]
            pattern = re.compile(r".*/%s.pdparams" %config_base)
            match_url_list = []
            for key in self.all_pdparams_urls:
                pdparams_url = str(key)
                md_filename_path = self.all_pdparams_urls[key]
                if pattern.match(pdparams_url):
                    match_url_list.append([md_filename_path, pdparams_url])

            if len(match_url_list) >0:
                self.all_model_info_list.append(PDAllModelInfo(str(match_url_list[0][0]), str(temp_file_name), str(match_url_list[0][1])))
            else:
                self.all_model_info_list.append(PDAllModelInfo('None', str(temp_file_name), 'None'))

        return self.all_model_info_list

    def get_filted_model_info_list(self):
        """
        use self.filter_data_file info to filter the self.all_model_info_list,
        then get the self.filted_model_info_list
        """
        if len(self.filted_model_info_list) > 0:
            return self.filted_model_info_list
        pass

        with open(self.filter_data_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                config_yaml = row[1]
                config_yaml = ''.join(config_yaml.split()) # remove all whitespace
                model_type = row[0]
                pattern = re.compile(r".*%s" %str(config_yaml))
                match_url_list = []
                for item in self.all_model_info_list:
                    if pattern.match(str(item.pdconfig)):
                        match_url_list.append([item.pdconfig, item.pdparams])

                if len(match_url_list) >0:
                    self.filted_model_info_list.append(PDFiltedModelInfo(str(model_type), str(config_yaml), str(match_url_list[0][1])))
                else:
                    self.filted_model_info_list.append(PDFiltedModelInfo(str(model_type), str(config_yaml), 'None'))

        return self.filted_model_info_list

    def download_models(self):
        """
        download models to models_download_path according to the self.download_mode.
        """
        if self.download_mode == 'None':
            return True
        elif self.download_mode == 'all':
            for item in self.all_model_info_list:
                if item.pdparams == 'None':
                    continue
                config_base = os.path.basename(item.pdconfig)
                config_base = os.path.splitext(config_base)[0]
                file_name = config_base + '.pdparams'
                self.download_pdparams_file(file_name, item.pdparams, self.models_download_path)
        elif self.download_mode == 'filtered':
            for item in self.filted_model_info_list:
                if item.pdparams == 'None':
                    continue
                config_base = os.path.basename(item.pdconfig)
                config_base = os.path.splitext(config_base)[0]
                file_name = config_base + '.pdparams'
                self.download_pdparams_file(file_name, item.pdparams, self.models_download_path)
        else:
            return False
        return True


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_project_dir =os.path.abspath(os.path.join(__dir__, '../../PaddleDetection'))
    default_filter_data_file =os.path.abspath(os.path.join(__dir__, '../data/paddledet.csv'))
    default_models_download_path =os.path.abspath(os.path.join(__dir__, './paddledet'))
    default_full_model_info_save_file =os.path.abspath(os.path.join(__dir__, './paddledet_full.csv'))
    default_filtered_model_info_save_file =os.path.abspath(os.path.join(__dir__, './paddledet_filtered.csv'))

    parser.add_argument("--project_dir", type=str, default=str(default_project_dir), help="PaddleDetection project directory")
    parser.add_argument("--filter_data_file", type=str, default=str(default_filter_data_file), help="The file we used to filter the model link")
    parser.add_argument("--download_mode", type=str, default='None', choices=['all', 'filtered', 'None'], help="Download mode, all: download all models, filtered: only download the filtered models, None: download nothing")
    parser.add_argument("--models_download_path", type=str, default=str(default_models_download_path), help="where to store the downloaded models")
    parser.add_argument("--full_model_info_save_file", type=str, default=str(default_full_model_info_save_file), help="The file to save the full model link info")
    parser.add_argument("--filtered_model_info_save_file", type=str, default=str(default_filtered_model_info_save_file), help="The file to save the filtered model link info")

    return parser.parse_args()

def convert_params_to_abspath(args):
    cwd = os.getcwd()
    if not os.path.isabs(args.project_dir):
        args.project_dir = os.path.abspath(os.path.join(cwd, args.project_dir))
    if not os.path.isabs(args.filter_data_file):
        args.filter_data_file = os.path.abspath(os.path.join(cwd, args.filter_data_file))
    if not os.path.isabs(args.models_download_path):
        args.models_download_path = os.path.abspath(os.path.join(cwd, args.models_download_path))
    if not os.path.isabs(args.full_model_info_save_file):
        args.full_model_info_save_file = os.path.abspath(os.path.join(cwd, args.full_model_info_save_file))
    if not os.path.isabs(args.filtered_model_info_save_file):
        args.filtered_model_info_save_file = os.path.abspath(os.path.join(cwd, args.filtered_model_info_save_file))

def welcome_info(args):
    logging.info("paddle detection models downloader begin.")
    logging.debug("args.project_dir: {}".format(args.project_dir))
    logging.debug("args.filter_data_file: {}".format(args.filter_data_file))
    logging.debug("args.download_mode: {}".format(args.download_mode))
    logging.debug("args.models_download_path: {}".format(args.models_download_path))
    logging.debug("args.full_model_info_save_file: {}".format(args.full_model_info_save_file))
    logging.debug("args.filtered_model_info_save_file: {}".format(args.filtered_model_info_save_file))

def main():
    #logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()
    convert_params_to_abspath(args)
    welcome_info(args)
    downloader = paddledet_downloader(args.project_dir, args.filter_data_file, args.download_mode, args.models_download_path, args.full_model_info_save_file, args.filtered_model_info_save_file)
    downloader.run()

if __name__ == "__main__":
    main()
    logging.info("paddle detection models downloader done.")
