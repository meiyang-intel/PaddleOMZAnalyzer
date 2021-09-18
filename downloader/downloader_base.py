import os
import logging
import sys
import csv
import re
from pathlib import Path
from bs4 import BeautifulSoup
import markdown
import requests
from abc import ABCMeta, abstractmethod
from collections import namedtuple
PDAllModelInfo=namedtuple('PDAllModelInfo', 'markdownfilename pdconfig pdparams')
PDFiltedModelInfo=namedtuple('PDFiltedModelInfo', 'modelname pdconfig pdparams')

class base_downloader(object):
    __metaclass__ = ABCMeta

    def __init__(self, project_dir, filter_data_file, download_mode, models_download_path, full_model_info_save_file, filtered_model_info_save_file):
        """
        init function
        """
        self.project_dir = project_dir
        self.filter_data_file = filter_data_file
        self.download_mode = download_mode
        self.models_download_path = models_download_path
        self.full_model_info_save_file = full_model_info_save_file
        self.filtered_model_info_save_file = filtered_model_info_save_file

        self.md_list = [] # all markdown file list
        self.all_pdparams_urls = {} #dict{url, mdfile}, use url as key to ensure it's unique
        self.all_yml_file_list = [] #all yml file list in project_dir directory
        self.all_yaml_file_list = [] #all yaml file list in project_dir directory
        self.all_model_info_list = [] #PDAllModelInfo list
        self.filted_model_info_list = [] #PDFiltedModelInfo list

    def get_all_markdown_file_list(self):
        """
        get all markdown file list in self.project_dir
        and store the list in self.md_list
        """
        if len(self.md_list) > 0:
            return self.md_list

        if not os.path.exists(self.project_dir):
            logging.critical('{} directory no exist.'.format(self.project_dir))
        else:
            self.md_list = Path(self.project_dir).glob('**/*.md')

        temp_list = []
        for mdfile in self.md_list:
            temp_list.append(mdfile)
        self.md_list = temp_list

        return self.md_list

    def get_all_yml_file_list(self):
        """
        get all yml file list in self.project_dir
        and store the list in self.all_yml_file_list
        """
        if len(self.all_yml_file_list) > 0:
            return self.all_yml_file_list

        if not os.path.exists(self.project_dir):
            logging.critical('{} directory no exist.'.format(self.project_dir))
        else:
            self.all_yml_file_list = Path(self.project_dir).glob('**/*.yml')

        temp_list = []
        for temp_file in self.all_yml_file_list:
            temp_list.append(temp_file)
        self.all_yml_file_list = temp_list

        return self.all_yml_file_list

    def get_all_yaml_file_list(self):
        """
        get all yaml file list in self.project_dir
        and store the list in self.all_yaml_file_list
        """
        if len(self.all_yaml_file_list) > 0:
            return self.all_yaml_file_list

        if not os.path.exists(self.project_dir):
            logging.critical('{} directory no exist.'.format(self.project_dir))
        else:
            self.all_yaml_file_list = Path(self.project_dir).glob('**/*.yaml')

        temp_list = []
        for temp_file in self.all_yaml_file_list:
            temp_list.append(temp_file)
        self.all_yaml_file_list = temp_list

        return self.all_yaml_file_list

    def get_all_pdparams_urls(self):
        """
        loop the self.md_list, find all pdparams link,
        create a dict{url, md_filename} list and store it to self.all_pdparams_urls
        """
        if len(self.all_pdparams_urls) > 0:
            return self.all_pdparams_urls

        if len(self.md_list) <= 0:
            logging.warning('md_list is empty!')
            return self.all_pdparams_urls

        for md_file in self.md_list:
            with open(md_file, 'r') as f:
                text = f.read()
                html_text = markdown.markdown(text)
                soup = BeautifulSoup(html_text, 'html.parser')
                tracks_pdparams = soup.find_all('a', attrs={'href': re.compile(r'\.pdparams$')})
                if len(list(tracks_pdparams))>0:
                    for track in tracks_pdparams:
                        pdparams_url = '{}'.format(track['href'])
                        self.all_pdparams_urls[pdparams_url] = md_file
                        # logging.debug('{}, {}'.format(pdparams_url, md_file))

        return self.all_pdparams_urls

    @abstractmethod
    def get_all_model_info_list(self):
        """
        match self.all_yml_file_list and self.all_yaml_file_list with self.all_pdparams_urls,
        then get the self.all_model_info_list

        """
        return self.all_model_info_list

    @abstractmethod
    def get_filted_model_info_list(self):
        """
        use self.filter_data_file info to filter the self.all_model_info_list,
        then get the self.filted_model_info_list
        """
        return self.filted_model_info_list


    def write_namedtuple_list_to_file(self, file_name, namedtuple_list):
        """
        write namedtuple_list:(PDAllModelInfo or PDFiltedModelInfo type)
        to file named file_name, if this file exist, will trunc it.
        PDAllModelInfo=namedtuple('PDAllModelInfo', 'markdownfilename pdconfig pdparams')
        PDFiltedModelInfo=namedtuple('PDFiltedModelInfo', 'modelname pdconfig pdparams')
        """
        #check list length
        if len(namedtuple_list) <= 0:
            logging.warning('The list you want to write is empty!')
            return False

        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(sorted(namedtuple_list))
        return True

    def download_pdparams_file(self, file_name, pdparams_url, model_save_path):
        """
        download file from pdparams_url link, and save it to model_save_path with named file_name
        """
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        logging.info('Downloading: {} from {}, save to {}'.format(file_name, pdparams_url, model_save_path))
        file_name = os.path.join(model_save_path, file_name)
        r = requests.get(pdparams_url, allow_redirects=True)
        with open(file_name, 'wb') as f:
            f.write(r.content)

    @abstractmethod
    def download_models(self):
        """
        download models to models_download_path according to the download_mode value.
        """
        return True

    def run(self):
        self.get_all_markdown_file_list()
        self.get_all_yml_file_list()
        self.get_all_yaml_file_list()
        self.get_all_pdparams_urls()
        self.get_all_model_info_list()
        self.get_filted_model_info_list()
        self.write_namedtuple_list_to_file(self.full_model_info_save_file, self.all_model_info_list)
        self.write_namedtuple_list_to_file(self.filtered_model_info_save_file, self.filted_model_info_list)
        self.download_models()
