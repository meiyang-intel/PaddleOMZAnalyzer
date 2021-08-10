# git clone https://github.com/PaddlePaddle/PaddleDetection
# iterately go through each markdown file to 
# - grab model config yaml and model pretrained pdparams file info from md file.
# - grab all yml file
# - then match pdparams to yml file

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup
import markdown

from common import PDModelInfo

class PaddleDetScraper(object):
    def __init__(self, homepage):
        self.homepage = homepage

    def __call__(self):
        paths = Path(self.homepage).glob('**/*.md')
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
        return all_md_urls

class PaddleDetFilter(object):
    def __init__(self, filter):
        self.filter = filter

    def __call__(self, all_md_urls, downdload=False):
        # collect model info of OMZ, cache to csv
        models = []
        with open(self.filter, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                config_yaml = row[1] # configs/.../*.yml
                config_yaml = ''.join(config_yaml.split()) # remove all whitespace'
                # print(config_yaml)

                cur_pdprams_url = ''

                # find the best matcher which is highly possible to be the correct config yml.
                for (config_url, pdprams_url) in all_md_urls:
                      if re.search(config_yaml+'$', config_url):
                        cur_pdprams_url = pdprams_url
                        models.append(PDModelInfo(row[0], config_yaml, cur_pdprams_url)) # possible more than one pdparams matches config yml , e.g. slim
                
                # second chance, to match basename only
                if not cur_pdprams_url:
                    config_base = os.path.basename(config_yaml)             
                    for (config_url, pdprams_url) in all_md_urls:
                        # print(config_yaml, config_url, re.match(config_yaml, config_url))
                        if re.search(config_base, config_url):
                            cur_pdprams_url = pdprams_url
                            models.append(PDModelInfo(row[0], config_yaml, cur_pdprams_url))            
               
                # if still fail, throw exception to check scrapy rules.
                if not cur_pdprams_url:
                    print('failed to get pdparams for {}, {}'.format(row[0], config_yaml))
                    continue        
        return models
        
def main(args):
    # scraper
    det_scaper = PaddleDetScraper(homepage=os.path.abspath(os.path.join(__dir__, '../../PaddleDetection')))   
    all_md_urls = det_scaper()
    with open('paddledet_full.csv', 'w', newline='') as csvfile: # cache urls for debugging
        headerList = ['pdconfig_url', 'pdparams_url']
        dw = csv.DictWriter(csvfile, delimiter=',', 
                            fieldnames=headerList)
        dw.writeheader()        
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(all_md_urls)

    # filter
    det_filter = PaddleDetFilter('../data/paddledet.csv')
    models = det_filter(all_md_urls)
    with open('paddledet_filtered.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(models)   

if __name__ == "__main__":
    import pandas
    from utils import parse_args
 
    args = parse_args()    
    main(args) 

    # display
    print(pandas.read_csv('paddledet_filtered.csv'))    