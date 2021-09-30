import argparse
import logging
import os
import re
import csv
import sys
from collections import namedtuple
__dir__ = os.path.dirname(os.path.abspath(__file__))
paddle_frontend_path =os.path.abspath(os.path.join(__dir__, '../analyzer'))
sys.path.insert(1, paddle_frontend_path)
from paddle_frontend import paddle_frontend_supported_ops
PDAnalyzerInfo=namedtuple('PDAnalyzerInfo', 'category total_models_downloadable total_models_exportable total_models_allops_converted total_ops_in_catogary total_ops_not_converted paddle_frontend_supported_ops_number unsupported_set')
PDExecutorInfo=namedtuple('PDExecutorInfo', 'category total_models_executable total_models_accuracycheck_pass')

ops_dict = {}
# def count_model_info_by_file(analyzer_file_name, category):
def manage_analyzer_result(analyzer_file_name, category):
    total_models_downloadable = 0
    total_models_exportable = 0
    total_models_allops_converted = 0
    sum_supported_ops_set = set()
    sum_unsupported_ops_set = set()

    analyzer_file_name = os.path.abspath(os.path.join(__dir__, '../analyzer/{}'.format(analyzer_file_name)))
    if not os.path.exists(analyzer_file_name):
        return PDAnalyzerInfo(category, total_models_downloadable, total_models_exportable,
                            total_models_allops_converted, len(sum_supported_ops_set), len(sum_unsupported_ops_set), len(paddle_frontend_supported_ops), 
                            sorted(sum_unsupported_ops_set))
        
    begin_pattern = re.compile(r"\".+")
    end_pattern = re.compile(r".+\"")
    with open(analyzer_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            pdparams_url = row[2]
            pdparams_url = ''.join(pdparams_url.split()) # remove all whitespace
            if pdparams_url != 'None':
                total_models_downloadable = total_models_downloadable + 1
                if row[-1] != 'ERROR' and row[3] != 'ERROR':
                    total_models_exportable = total_models_exportable + 1
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
                                operate_name = row[i].replace('"', '')
                                sum_unsupported_ops_set.add(operate_name)
                                if operate_name not in ops_dict:
                                    ops_dict[operate_name] = [pdparams_url]
                                else:
                                    ops_dict[operate_name].append(pdparams_url)
                                i = i + 1
                        operate_name = row[i].replace('"', '')
                        sum_unsupported_ops_set.add(operate_name)
                        if operate_name not in ops_dict:
                            ops_dict[operate_name] = [pdparams_url]
                        else:
                            ops_dict[operate_name].append(pdparams_url)
                    else:
                        total_models_allops_converted = total_models_allops_converted + 1

        logging.debug(category, total_models_downloadable, total_models_exportable, total_models_allops_converted, len(sum_supported_ops_set), len(sum_unsupported_ops_set), len(paddle_frontend_supported_ops), sum_unsupported_ops_set)
        return PDAnalyzerInfo(category, total_models_downloadable, total_models_exportable, total_models_allops_converted, len(sum_supported_ops_set), len(sum_unsupported_ops_set), len(paddle_frontend_supported_ops), sorted(sum_unsupported_ops_set))


def manage_executor_result(executor_file_name, category):
    total_models_executable = 0
    total_models_accuracycheck_pass = 0

    executor_file_name = os.path.abspath(os.path.join(__dir__, '../executor/{}'.format(executor_file_name)))
    if not os.path.exists(executor_file_name):
        return PDExecutorInfo(category, total_models_executable, total_models_accuracycheck_pass)

    with open(executor_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            wormup_time = row[-1]
            infer_time = row[-2]
            status = row[-5]
            if wormup_time != 'NONE' and infer_time != 'NONE':
                total_models_executable = total_models_executable + 1
                if status == 'infer accuracy pass':
                    total_models_accuracycheck_pass = total_models_accuracycheck_pass + 1
        logging.debug(category, total_models_executable, total_models_accuracycheck_pass)
        return PDExecutorInfo(category, total_models_executable, total_models_accuracycheck_pass)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--category", type=str, default='', choices=['classify', 'segmentation', 'detection'], help="select the category of models type, use to output table title of result, default(None)")
    parser.add_argument("--analyzer_result_file", type=str, default='', help="The analyzer result file you want to analytic, default(None)")
    parser.add_argument("--executor_result_file", type=str, default='', help="The executor result file you want to analytic, default(None)")
    return parser.parse_args()

def get_default_count_list():
    default_list = [
        ('classify',        'paddleclas_full_operators.csv',    'paddleclas_full_result.csv'),
        ('segmentation',    'paddleseg_full_operators.csv',     'paddleseg_full_result.csv'),
        ('detection',       'paddledet_full_operators.csv',     'paddledet_full_result.csv'),
        ('nlp',             'paddlenlp_filtered_operators.csv', 'paddlenlp_filtered_result.csv'),
        ('ocr',             'paddleocr_filtered_operators.csv', 'paddleocr_filtered_result.csv')
    ]
    return default_list

def main():
    #logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname)s]-[%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    args = parse_args()

    result_table_one = []
    result_table_one.append(['category', 'total_models_downloadable', 'total_models_exportable', 'total_models_allops_converted', 'total_models_executable', 'total_models_accuracycheck_pass', 'total_models_accuracycheck_pass/total_models_exportable'])
    result_table_two = []
    result_table_two.append(['category', 'total_ops_in_catogary', 'total_ops_not_converted', 'paddle_frontend_supported_ops_number', 'total_ops_in_catogary/total_ops_not_converted', 'unsupported_set',''])
    ops_result_table = []

    if args.analyzer_result_file != '' and args.category != '':
        logging.info("Single analyze mode:")
        logging.info("{}: {} , {}.".format(args.category, args.analyzer_result_file, args.executor_result_file))

        default_list = [(args.category, args.analyzer_result_file, args.executor_result_file)]
    else:
        logging.info("You did not set the analyzer_result_file or executor_result_file or category, will run default analyze mode:")
        default_list = get_default_count_list()
        if len(default_list) == 0:
            logging.error("get default_list failed.")
            return
            
    for category, analyzer_file, executor_file in default_list:
        logging.info("{}: {}, {}.".format(category, analyzer_file, executor_file))
        result_analyzer = manage_analyzer_result(analyzer_file, category)
        result_executor = manage_executor_result(executor_file, category)
        result_table_one.append([result_analyzer.category, result_analyzer.total_models_downloadable, result_analyzer.total_models_exportable, result_analyzer.total_models_allops_converted, result_executor.total_models_executable, result_executor.total_models_accuracycheck_pass, str(result_executor.total_models_accuracycheck_pass) + '/' + str(result_analyzer.total_models_exportable)])
        result_table_two.append([result_analyzer.category, result_analyzer.total_ops_in_catogary, result_analyzer.total_ops_not_converted, result_analyzer.paddle_frontend_supported_ops_number, str(result_analyzer.total_ops_in_catogary) + '/' + str(result_analyzer.total_ops_not_converted), result_analyzer.unsupported_set, ''])

    with open('./report.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(result_table_one)
        writer.writerow('')
        writer.writerows(result_table_two)

    for key, value in ops_dict.items():
        ops_result_table.append([key, len(value), value])

    with open('./report_ops.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['unsupported_op_name', 'total_models', 'models_list'])
        writer.writerows(sorted(ops_result_table))

if __name__ == "__main__":
    main()
