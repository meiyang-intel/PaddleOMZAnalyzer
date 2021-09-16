import pytest
from executor_class import performance_and_accuracy_test_by_params
from collections import namedtuple
import os
import csv
from pathlib import Path

def create_params_list_by_csv_file(csv_file, model_category_path, result_save_file, batch_size=1, warmup=0, repeats=1, openvino_api_type='sync'):
    Params=namedtuple('Params', 'result_prefix_str, model_file, model_params_file, result_save_file, result_level, batch_size, warmup, repeats, openvino_api_type')

    params_list = []
    if not os.path.exists(csv_file) or not os.path.exists(model_category_path):
        print('Have no this {} directory or {} not exist.'.format(model_category_path, csv_file))
        return [False]

    #delete result_save_file
    if os.path.exists(result_save_file):
        os.remove(result_save_file)

    result_prefix_str = str('')
    # loop get params_list
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            result_level = 1 if row[-1] == 'None' else 0  # based on anaylyzer report
            result_prefix_str = row[0] + ',' + row[1] + ',' + row[2]
            config_yaml = row[1]
            config_yaml = ''.join(config_yaml.split()) # remove all whitespace
            config_base = os.path.basename(config_yaml)
            config_base = os.path.splitext(config_base)[0]

            exported_path = os.path.abspath(os.path.join(model_category_path, config_base))

            if os.path.exists(exported_path):
                p = Path(exported_path)
                pdmodel = p.glob('**/*.pdmodel')
                pdiparams = p.glob('**/*.pdiparams')

                pdmodel_list = []
                pdiparams_list = []
                for path in pdmodel:
                   pdmodel_list.append(path)
                for path in pdiparams:
                   pdiparams_list.append(path)

                params_list.append(Params(result_prefix_str, None if len(pdmodel_list) != 1 else pdmodel_list[0], None if len(pdiparams_list) != 1 else pdiparams_list[0], result_save_file, result_level, batch_size, warmup, repeats, openvino_api_type))
            else:
                params_list.append(Params(result_prefix_str, None, None, result_save_file, result_level, batch_size, warmup, repeats, openvino_api_type))

    return params_list

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddleclas_operators.csv', '../exporter/paddleclas', './paddleclas_result.csv', warmup=1))
def get_param_of_classify_and_test(request):
    return request.param

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddleseg_operators.csv', '../exporter/paddleseg', './paddleseg_result.csv', warmup=1))
def get_param_of_segmentation_and_test(request):
    return request.param

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddledet_operators.csv', '../exporter/paddledet', './paddledet_result.csv', warmup=1))
def get_param_of_detection_and_test(request):
    return request.param

def test_classify(get_param_of_classify_and_test):
    param = get_param_of_classify_and_test
    print('######Classify Models Test######')
    print("result_prefix_str:", param.result_prefix_str)
    print("model_file:", param.model_file)
    print("model_params_file:", param.model_params_file)
    print("result_save_file:", param.result_save_file)
    print("result_level:", param.result_level)
    print("batch_size:", param.batch_size)
    print("warmup:", param.warmup)
    print("repeats:", param.repeats)
    print("openvino_api_type:", param.openvino_api_type)
    performance_and_accuracy_test_by_params(param.result_prefix_str, param.model_file, param.model_params_file, param.result_save_file,  param.result_level, param.batch_size, param.warmup, param.repeats, param.openvino_api_type)
    return

def test_segmentation(get_param_of_segmentation_and_test):
    param = get_param_of_segmentation_and_test
    print('######Segmentation Models Test######')
    print("result_prefix_str:", param.result_prefix_str)
    print("model_file:", param.model_file)
    print("model_params_file:", param.model_params_file)
    print("result_save_file:", param.result_save_file)
    print("result_level:", param.result_level)
    print("batch_size:", param.batch_size)
    print("warmup:", param.warmup)
    print("repeats:", param.repeats)
    print("openvino_api_type:", param.openvino_api_type)
    performance_and_accuracy_test_by_params(param.result_prefix_str, param.model_file, param.model_params_file, param.result_save_file,  param.result_level, param.batch_size, param.warmup, param.repeats, param.openvino_api_type)
    return

def test_detection(get_param_of_detection_and_test):
    param = get_param_of_detection_and_test
    print('######Detection Models Test######')
    print("result_prefix_str:", param.result_prefix_str)
    print("model_file:", param.model_file)
    print("model_params_file:", param.model_params_file)
    print("result_save_file:", param.result_save_file)
    print("result_level:", param.result_level)
    print("batch_size:", param.batch_size)
    print("warmup:", param.warmup)
    print("repeats:", param.repeats)
    print("openvino_api_type:", param.openvino_api_type)
    performance_and_accuracy_test_by_params(param.result_prefix_str, param.model_file, param.model_params_file, param.result_save_file,  param.result_level, param.batch_size, param.warmup, param.repeats, param.openvino_api_type)
    return

if __name__ == '__main__':
    # pytest.main(['-s','pytest_executor.py'])
    pytest.main(['pytest_executor.py'])
