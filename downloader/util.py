import os

import re

from collections import namedtuple
PDModelInfo=namedtuple('PDModelInfo', 'modelname pdconfig pdparams')

import requests
from bs4 import BeautifulSoup
def scrape_pdparams(vgm_url):
    html_text = requests.get(vgm_url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    attrs = {
        'href': re.compile(r'\.pdparams$')
    }

    tracks = soup.find_all('a', attrs=attrs, string=re.compile(r'^((?!\().)*$'))
    return tracks

def download_pdparams(pdprams_url, download_folder):   
    #
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    #
    pdprams_name = os.path.basename(pdprams_url)
    pdprams_name = os.path.splitext(pdprams_name)[0]    

    file_name = '{}.pdparams'.format(pdprams_name)
    file_name = os.path.join(download_folder, file_name)

    # Print to the console to keep track of how the scraping is coming along.
    print('Downloading: {} {}'.format(file_name, pdprams_url))    

    # Download the track
    r = requests.get(pdprams_url, allow_redirects=True)
    with open(file_name, 'wb') as f:
        f.write(r.content)