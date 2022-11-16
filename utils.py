import numpy as np
import logging
import os
import re

def str2bool(v):
    return v.lower() in ("true", "t", "1")

'''
paddle specific helpers
'''
import paddle

PADDLE_FLOAT_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64"
}
PADDLE_INT_DICT = {
    paddle.int8: "int8",
    paddle.int32: "int32",
    paddle.int64: "int64"    
}
PADDLE_DTYPE_DICT = {**PADDLE_FLOAT_DICT, **PADDLE_INT_DICT}

def is_float_tensor(dtype):
    """Is a float tensor"""
    return dtype in PADDLE_FLOAT_DICT.keys()

def get_tensor_dtype(dtype):
    assert dtype in PADDLE_DTYPE_DICT.keys()
    return PADDLE_DTYPE_DICT[dtype]

def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)    

def check_s_file_exists(filename, tag):
    size = tag["end"] - tag["start"] + 1
    try:
        file_stat = os.stat(filename)
        if file_stat.st_size == size:
            return True
    except:
        pass
    return False

import requests
def run_download(url, filename, tag, retry):
    try:
        s_filename = filename + '.' + tag["range_str"]
        if check_s_file_exists(s_filename, tag):
            return True

        headers = { 'Range' : tag["range_str"] }
        while retry:
            logging.debug('[{}] Trying to download {}'.format(retry, s_filename))
            try:
                r = requests.get(url, stream = True, headers = headers)
                with open(s_filename, 'wb') as f:
                    f.write(r.content)
                if check_s_file_exists(s_filename, tag):
                    break
            except:
                logging.debug('Download error on {}, try again'.format(s_filename))
                pass
            retry -= 1
        if retry > 0:
            logging.debug('Download complete on {}'.format(s_filename))
            return True
        else:
            logging.error('Download error on {}'.format(s_filename))
            return False
    except:
        logging.error('Download error on {}'.format(s_filename))
        return False

def download(url, filename, split_len=8*1024*1024, workers_number=10, retry=10):
    """
    Download tools to split large file into pieces and download
    """
    from multiprocessing import Pool

    support_split = False
    headers = {
        'Range': 'bytes=0-4'
    }
    try:
        r = requests.head(url, headers = headers)
        crange = r.headers['content-range']
        total_size = int(re.match('^bytes 0-4/(\d+)$', crange).group(1))
        logging.debug(total_size)
        support_split = True
    except:
        pass

    if not support_split:
        logging.debug('Download with content-range is not supported for {}.'.format(filename))
        while retry:
            try:
                r = requests.get(url, allow_redirects=True)
                with open(file_name, 'wb') as f:
                    f.write(r.content)
                logging.info('Download {} complete.'.format(filename))
                return
            except:
                logging.debug('Download error on {}, try again'.format(s_filename))
                retry -= 1
        logging.error('Download {} incomplete'.format(filename))
        return

    #total_size = int(r.headers['Content-Length'])
    tags = []
    start_pos = 0
    while start_pos < total_size:
        size = start_pos + split_len
        if size >= total_size:
            end_pos = total_size - 1
        else:
            end_pos = size - 1

        tag = {
            "start" : start_pos,
            "end"   : end_pos,
            "range_str": 'bytes={}-{}'.format(start_pos, end_pos)
        }
        logging.debug(tag)
        tags.append(tag)
        start_pos = size

    p = Pool(workers_number)

    args = [(url ,filename, tags[i], retry) for i in range(0, len(tags))]
    results = p.starmap(run_download, args)
    logging.debug('{}: {}'.format(filename, results))
    complete = True
    for result in results:
        if not result:
            complete = False
            break

    if complete:
        logging.info('Download {} complete.'.format(filename))
        # delete the temporary files
        with open(filename, 'wb') as f:
            for tag in tags:
                s_filename = filename + '.' + tag["range_str"]
                with open(s_filename, 'rb') as sf:
                    f.write(sf.read())
                os.remove(s_filename)
    else:
        logging.error('Download {} incomplete'.format(filename))

def md_table_to_dict(md_file):
    """
    Convert tables in markdown file to dict
    """
    with open(md_file, 'r') as f:
        content = f.read()

    colunms = 0
    tables = []
    is_table = False
    pre_items = []
    pre_columns = 0
    table = []
    # store all the table like rows
    for row in content.split('\n'):
        items = [item.strip() for item in row.split('|')[1:-1]]
        columns = len(items)
        if columns > 1 and pre_columns == columns:
            # detect the table head and insert it to table
            if not is_table:
                table.append(pre_items)
            table.append(items)
            is_table = True
        elif is_table:
            tables.append(table)
            logging.debug(table)
            table = []
            is_table = False
        pre_items = items
        pre_columns = columns

    # retrieve table with head and convert it to dict
    dicts = []
    for table in tables:
        second_line_flag = all([ None != re.fullmatch(':?---+:?', i) for i in table[1] ])
        if second_line_flag:
            table_dict = []
            head = table[0]
            for row in table[2:]:
                table_dict.append({head[i] : row[i] for i in range(len(row))})
            logging.debug(table_dict)
            dicts.append(table_dict)

    return dicts

def md_link_to_dict(text):
    """
    Convert all the links in markdown text to dict
    """
    matches = re.compile('\[(.+?)\]\((https?://.+?)\)').findall(text)
    return {m[0] : m[1] for m in matches}

if __name__ == '__main__':
    #print(md_table_to_dict("/home/iot/Working/PaddleOCR/README.md"))
    print(md_link_to_dict("[inference model](http://afdsadsf) / [ggg](https://afdasfasdfadsf)"))
