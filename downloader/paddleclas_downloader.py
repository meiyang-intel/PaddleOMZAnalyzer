import re
import csv
import os
import sys

from common import PDModelInfo
from downloader_helper import download_pdparams, scrape_pdparams

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))

    # scrapy linker of *.pdparams
    tracks = scrape_pdparams('https://github.com/PaddlePaddle/PaddleClas/blob/release/2.2/docs/zh_CN/models/models_intro.md')

    # collect model info of OMZ, cache to csv
    models = []
    with open('../data/paddleclas.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]
            pdprams_url = ''

            # find the best matcher which is highly possible to be the correct pdparams.
            for track in tracks:
                # search title for the first chance
                track_title = track.text.strip().replace('/', '-')
                if re.match(config_base+'$', track_title):
                    pdprams_url = '{}'.format(track['href'])
                    #print(config_yaml, track_title, pdprams_url)
                    break
            
            if not pdprams_url:
                # search title for second chance
                for track in tracks:
                    track_url = '{}'.format(track['href'])
                    #print(config_base, track_url)
                    if re.search(config_base, track_url):
                        pdprams_url = '{}'.format(track['href'])
                        break
            
            # if still fail, throw exception to check scrapy rules.
            if not pdprams_url:
                print('failed to get pdparams for {} {}'.format(row[0], config_yaml))                             

            models.append(PDModelInfo(row[0], config_yaml, pdprams_url))

    with open('paddleclas_full.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(models)

    # download
    if len(sys.argv) > 1:
        for m in models:
            abspath = os.path.abspath(__file__)
            dirs, _ = os.path.split(abspath)
            download_pdparams(m.pdparams, f'{dirs}/paddleclas')

