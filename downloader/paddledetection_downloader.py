# git clone https://github.com/PaddlePaddle/PaddleDetection
# iterately go through each markdown file to 
# - grab model config yaml and model pretrained pdparams file info from md file.
# - grab all yml file
# - then match pdparams to yml file

import re
import csv
import os
import sys

import pandas as pd

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

    all_md_urls = set(()) # use set instead of list to avoid duplicate item
    count_pdparams = 0
    for path in paths:      
        with open(path, 'r') as f:
            text = f.read()
            html_text = markdown.markdown(text)

            soup = BeautifulSoup(html_text, 'html.parser')

            tracks_pdparams = soup.find_all('a', attrs={'href': re.compile(r'\.pdparams$')}, string=re.compile(r'^((?!\().)*$'))            

            if len(list(tracks_pdparams))>0:
                # debugging
                tracks_ymls = soup.find_all('a', attrs={'href': re.compile(r'\.yml$|\.yaml$')}, string=re.compile(r'^((?!\().)*$')) # either yml or yaml   
                print(path, len(list(tracks_pdparams)), len(tracks_ymls))
                count_pdparams += len(list(tracks_pdparams))

                for track in tracks_pdparams:
                    track_config = track.findNext('a', attrs={'href': re.compile(r'\.yml$|\.yaml$')}, string=re.compile(r'^((?!\().)*$'))
                    if track_config is None:
                        continue # ignore

                    configs_url = '{}'.format(track_config['href'])
                    pdparams_url = '{}'.format(track['href'])
                    all_md_urls.add((configs_url, pdparams_url))

    print(len(all_md_urls), count_pdparams)
    with open('paddledet_urls.csv', 'w', newline='') as csvfile: # cache urls for debugging
        headerList = ['pdconfig_url', 'pdparams_url']
        dw = csv.DictWriter(csvfile, delimiter=',', 
                            fieldnames=headerList)
        dw.writeheader()        
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(all_md_urls)

    # display
    print(pd.read_csv('paddledet_urls.csv'))

    # # read column
    # col_list = ["mdfile", "pdparams_url"]
    # df = pd.read_csv("paddledet_urls.csv", usecols=col_list)
    # print(df["mdfile"])
    # print(type(df), type(df['mdfile']))


    # import pandas as pd

    # titanic_data = pd.read_csv(r'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    # print(titanic_data.head())


      

    # # go matcher
    # for pdparams_url in all_pdparams_urls:
    #     pdprams_basename = os.path.basename(pdparams_url)
    #     pdprams_basename = os.path.splitext(pdprams_basename)[0]
    #     #print(pdprams_basename)
    
    # for configs_url in all_configs_urls:
    #     config_basename = os.path.basename(configs_url)
    #     config_basename = os.path.splitext(config_basename)[0]
    #     #print(config_basename)