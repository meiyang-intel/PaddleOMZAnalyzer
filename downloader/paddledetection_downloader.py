# git clone https://github.com/PaddlePaddle/PaddleDetection
# iterately go through each markdown file to 
# - grab model config yaml and model pretrained pdparams file info from md file.
# - grab all yml file
# - then match pdparams to yml file

import re
import csv
import os
import sys

#import glob
from pathlib import Path

import markdown

import requests
from bs4 import BeautifulSoup

from common import PDModelInfo
from downloader_helper import download_pdparams, scrape_pdparams

from collections import namedtuple
MDURLInfo=namedtuple('MDURLInfo', 'mdfile pdconfig pdparams')

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))

    paddledet_folder = os.path.abspath(os.path.join(__dir__, '../../PaddleDetection'))
    print(paddledet_folder)

    paths = Path(paddledet_folder).glob('**/*.md')
    #print(len(list(paths))) # NOTE: in-place list op

    all_md_urls = []

    for path in paths:
        # print(path)        
        with open(path, 'r') as f:
            text = f.read()
            html_text = markdown.markdown(text)

            soup = BeautifulSoup(html_text, 'html.parser')

            tracks_pdparams = soup.find_all('a', attrs={'href': re.compile(r'\.pdparams$')}, string=re.compile(r'^((?!\().)*$'))
            tracks_configs = soup.find_all('a', attrs={'href': re.compile(r'\.yml$')}, string=re.compile(r'^((?!\().)*$'))

            # print(path, len(list(tracks_pdparams)), len(list(tracks_configs)))

            if len(list(tracks_pdparams))>0 and len(list(tracks_configs))>0:
                pdparams_urls = set(()) # use set instead of list to avoid duplicate item
                configs_urls = set(())
                for track in tracks_pdparams:
                    pdparams_url = '{}'.format(track['href'])
                    pdparams_urls.add(pdparams_url)

                for track in tracks_configs:
                    configs_url = '{}'.format(track['href'])
                    configs_urls.add(configs_url)
                
                # print(path, len(configs_urls), len(pdparams_urls))

                # example: 
                # https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_ghostnet.pdparams 
                # https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssd_vgg16_512_voc.yml

                for idx in range(min(len(configs_urls), len(pdparams_urls))):
                    all_md_urls.append(MDURLInfo(path.name, configs_urls.pop(), pdparams_urls.pop()))
  
    print(len(all_md_urls))
    with open('paddledet_urls.csv', 'w', newline='') as csvfile: # cache urls for debugging
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(all_md_urls)

    # # go matcher
    # for pdparams_url in all_pdparams_urls:
    #     pdprams_basename = os.path.basename(pdparams_url)
    #     pdprams_basename = os.path.splitext(pdprams_basename)[0]
    #     #print(pdprams_basename)
    
    # for configs_url in all_configs_urls:
    #     config_basename = os.path.basename(configs_url)
    #     config_basename = os.path.splitext(config_basename)[0]
    #     #print(config_basename)