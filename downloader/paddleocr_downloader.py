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
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from utils import md_table_to_dict, md_link_to_dict

class paddleocr_downloader(base_downloader):
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

    def get_all_pdparams_urls(self):
        """
        loop the self.md_list, find all model links,
        create a dict{url, md_filename} list and store it to self.all_pdparams_urls
        """
        if len(self.all_pdparams_urls) > 0:
            return self.all_pdparams_urls

        if len(self.md_list) <= 0:
            logging.warning('md_list is empty!')
            return self.all_pdparams_urls

        for md_file in self.md_list:
            dicts = md_table_to_dict(md_file)
            for table in dicts:
                for row in table:
                    infer_url = None
                    for key, value in row.items():
                        links = md_link_to_dict(value)
                        for k, v in links.items():
                            if k == 'inference model':
                                infer_url = v
                    if infer_url:
                        self.all_pdparams_urls[infer_url] = md_file
                        logging.debug('{}, {}'.format(infer_url, md_file))
        return self.all_pdparams_urls

    def get_all_model_info_list(self):
        """
        match self.all_yml_file_list and self.all_yaml_file_list with self.all_pdparams_urls,
        then get the self.all_model_info_list
        """
        self.all_model_info_list = []
        for url, md_file in self.all_pdparams_urls.items():
            self.all_model_info_list.append(PDAllModelInfo(md_file, 'None', url))

        return self.all_model_info_list

    def get_filted_model_info_list(self):
        """
        use self.filter_data_file info to filter the self.all_model_info_list,
        then get the self.filted_model_info_list
        """
        if len(self.filted_model_info_list) > 0:
            return self.filted_model_info_list

        with open(self.filter_data_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            match_url_list = []
            for row in reader:
                model_type = row[0]
                model_type = ''.join(model_type.split()) # remove all whitespace
                pattern = re.compile(r".*?%s.*?_infer.tar" %str(model_type))
                for item in self.all_model_info_list:
                    if pattern.match(str(item.pdparams)):
                        match_url_list.append([None, item.pdparams])
                logging.debug(match_url_list)

            self.filted_model_info_list = [PDFiltedModelInfo(str(model_type), 'None', str(match_url_list[i][1])) for i in range(len(match_url_list))]

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
                file_name = os.path.basename(item.pdparams)
                self.download_pdparams_file(file_name, item.pdparams, self.models_download_path)
        elif self.download_mode == 'filtered':
            for item in self.filted_model_info_list:
                if item.pdparams == 'None':
                    continue
                file_name = os.path.basename(item.pdparams)
                self.download_pdparams_file(file_name, item.pdparams, self.models_download_path)
        else:
            return False

        return True


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_project_dir =os.path.abspath(os.path.join(__dir__, '../../PaddleOCR'))
    default_filter_data_file =os.path.abspath(os.path.join(__dir__, '../data/paddleocr.csv'))
    default_models_download_path =os.path.abspath(os.path.join(__dir__, './paddleocr'))
    default_full_model_info_save_file =os.path.abspath(os.path.join(__dir__, './paddleocr_full.csv'))
    default_filtered_model_info_save_file =os.path.abspath(os.path.join(__dir__, './paddleocr_filtered.csv'))

    parser.add_argument("--project_dir", type=str, default=str(default_project_dir), help="PaddleSeg project directory")
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
    logging.info("paddle segmention models downloader begin.")
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
    downloader = paddleocr_downloader(args.project_dir, args.filter_data_file, args.download_mode, args.models_download_path, args.full_model_info_save_file, args.filtered_model_info_save_file)
    downloader.run()

if __name__ == "__main__":
    main()
    logging.info("paddle segmention models downloader done.")
