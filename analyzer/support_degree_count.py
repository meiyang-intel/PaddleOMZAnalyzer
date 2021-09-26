import argparse
import logging
import os
import re
import csv
from paddle_frontend import paddle_frontend_supported_ops
from collections import namedtuple
PDModelInfo=namedtuple('PDModelInfo', 'category sum_link_number sum_export_success_number all_supported_number sum_supported_ops_number sum_unsupported_ops_number paddle_frontend_supported_ops_number unsupported_set')

__dir__ = os.path.dirname(os.path.abspath(__file__))
def count_model_info_by_file(analyzer_file_name, category):
    sum_link_number = 0
    sum_export_success_number = 0
    all_supported_number = 0
    sum_supported_ops_set = set()
    sum_unsupported_ops_set = set()

    begin_pattern = re.compile(r"\".+")
    end_pattern = re.compile(r".+\"")
    with open(analyzer_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            pdparams_url = row[2]
            pdparams_url = ''.join(pdparams_url.split()) # remove all whitespace
            if pdparams_url != 'None':
                sum_link_number = sum_link_number + 1
                if row[-1] != 'ERROR' and row[3] != 'ERROR':
                    sum_export_success_number = sum_export_success_number + 1
                    i = 3
                    if begin_pattern.match(row[i]):
                        while not end_pattern.match(row[i]):
                            sum_supported_ops_set.add(row[i].replace('"', ''))
                            i = i + 1
                    sum_supported_ops_set.add(row[i].replace('"', ''))
                    i = i + 1
                    if row[-1] != "None":
                        if begin_pattern.match(row[i]):
                            while not end_pattern.match(row[i]):
                                sum_unsupported_ops_set.add(row[i].replace('"', ''))
                                i = i + 1
                        sum_unsupported_ops_set.add(row[i].replace('"', ''))
                    else:
                        all_supported_number = all_supported_number + 1

        logging.debug(category, sum_link_number, sum_export_success_number, all_supported_number, len(sum_supported_ops_set), len(sum_unsupported_ops_set), len(paddle_frontend_supported_ops), sum_unsupported_ops_set)
        return PDModelInfo(category, sum_link_number, sum_export_success_number, all_supported_number, len(sum_supported_ops_set), len(sum_unsupported_ops_set), len(paddle_frontend_supported_ops), sum_unsupported_ops_set)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--analyzer_file", type=str, default='', help="The analyzer result file you want to analytic, default(None)")
    parser.add_argument("--category", type=str, default='', choices=['classify', 'segmentation', 'detection'], help="select the category of models type, use to output table title of result, default(None)")
    return parser.parse_args()

def get_default_count_list():
    default_list = [] 
    default_category = 'classify'
    default_file_path =os.path.abspath(os.path.join(__dir__, './paddleclas_full_operators.csv'))
    if os.path.exists(default_file_path):
        default_list.append((default_category, default_file_path))
    else:
        logging.warning("{}: have no this file {}, jump it.".format(default_category, default_file_path))

    default_category = 'segmentation'
    default_file_path =os.path.abspath(os.path.join(__dir__, './paddleseg_full_operators.csv'))
    if os.path.exists(default_file_path):
        default_list.append((default_category, default_file_path))
    else:
        logging.warning("{}: have no this file {}, jump it.".format(default_category, default_file_path))
    
    default_category = 'detection'
    default_file_path =os.path.abspath(os.path.join(__dir__, './paddledet_full_operators.csv'))
    if os.path.exists(default_file_path):
        default_list.append((default_category, default_file_path))
    else:
        logging.warning("{}: have no this file {}, jump it.".format(default_category, default_file_path))

    return default_list


def main():
    #logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()
    if args.analyzer_file != '' and args.category != '':
        logging.info("Single analyze mode:")
        logging.info("{}: {}.".format(args.category, args.analyzer_file))
        if not os.path.exists(default_file_path):
            logging.error("{} not exist.".format(default_file_path))
            return
        result = count_model_info_by_file(args.analyzer_file, args.category)
        logging.info("result:", result.category, result.sum_link_number, result.sum_export_success_number, result.all_supported_number, result.sum_supported_ops_number, result.sum_unsupported_ops_number, result.paddle_frontend_supported_ops_number, result.unsupported_set)
    else:
        logging.info("You did not set the analyzer_file or category, will run default analyze mode:")
        default_list = get_default_count_list()
        if len(default_list) == 0:
            logging.error("get default_list failed.")
            return
        result_list = []
        for category, analyzer_file in default_list:
            logging.info("{}: {}.".format(category, analyzer_file))
            result = count_model_info_by_file(analyzer_file, category)
            result_list.append(result)

        with open('./models_support_degree_count.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['category', 'sum_link_number', 'sum_export_success_number', 'all_supported_number', 'sum_supported_ops_number', 'sum_unsupported_ops_number', 'paddle_frontend_supported_ops_number',  'unsupported_set'])
            writer.writerows(sorted(result_list))

if __name__ == "__main__":
    main()
