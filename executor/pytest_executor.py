import pytest
from collections import namedtuple
import os
import csv
import argparse
from pathlib import Path

from executor import OpenvinoExecutor, PaddlenlpPredictor, PaddleExecutor, PaddlePredictorExecutor, compare
def performance_and_accuracy_test_by_params(result_prefix_str, model_file, model_params_file, result_save_file,  result_level=1, batch_size=1, warmup=0, repeats=1, openvino_api_type: str = 'sync'):
    '''
    'result_level': 0: analyzer failed
                    1: analyzer success
                    2: openvino inference success
                    3: paddle inference result equal to the result of openvino inference
    writefile row: 'result_prefix_str result_level paddle_warmup_frame_time paddle_repeats_per_frame_time openvino_warmup_frame_time openvino_repeats_per_frame_time'
    '''
    # ResultInfo=namedtuple('ResultInfo', 'result_prefix_str result_level paddle_first_frame_time paddle_repeats_per_frame_time openvino_first_frame_time openvino_repeats_per_frame_time')
    result_level_dict = dict(zip(range(0,4),
                                ['op mapper failed', 'all ops mapped', 'openvino infered success', 'infer accuracy pass']))

    try:
        paddle_warmup_frame_time = None
        paddle_repeats_per_frame_time = None
        openvino_warmup_frame_time = None
        openvino_repeats_per_frame_time = None
        # No file or multi file
        if model_file == None or model_params_file == None:
            return

        # create test data
        pdpd_executor = PaddleExecutor(model_file)
        test_inputs = pdpd_executor.generate_inputs(batch_size)

        ## paddle predictor inference
        pdpd_predict_executor = PaddlePredictorExecutor(str(model_file), str(model_params_file))
        pdpd_predict_executor.run(test_inputs, warmup, repeats)
        pdpd_predict_result = pdpd_predict_executor.get_inference_results()
        paddle_warmup_frame_time = pdpd_predict_executor.warmup_time
        paddle_repeats_per_frame_time = pdpd_predict_executor.repeat_time

        if result_level<=0:
            return

        ## openvino inference
        ov_executor = OpenvinoExecutor(model_file, 10, openvino_api_type)
        ov_executor.run(test_inputs, warmup, repeats)
        ov_result = ov_executor.get_inference_results()
        openvino_warmup_frame_time = ov_executor.warmup_time
        openvino_repeats_per_frame_time = ov_executor.repeat_time
        result_level = 2

        # compare
        ov_result = [ov_result[k] for k in sorted(ov_result)]
        pdpd_predict_result = [pdpd_predict_result[k] for k in sorted(pdpd_predict_result)]
        res = compare(ov_result, pdpd_predict_result)

        if res:
            result_level = 3

    finally:
        timestamps = [paddle_warmup_frame_time, paddle_repeats_per_frame_time, openvino_warmup_frame_time, openvino_repeats_per_frame_time]
        print(timestamps)
        timestamps = ["{:.2f}".format(t) if t is not None else 'NONE' for t in timestamps]
        # write append to result_save_file
        with open(result_save_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([result_prefix_str, result_level_dict[result_level]] + timestamps)


"""
pytest test cases
"""

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

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddleclas_full_operators.csv', '../exporter/paddleclas', './paddleclas_full_result.csv', warmup=1))
def get_param_of_all_classify_and_test(request):
    return request.param

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddleseg_full_operators.csv', '../exporter/paddleseg', './paddleseg_full_result.csv', warmup=1))
def get_param_of_all_segmentation_and_test(request):
    return request.param

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddledet_full_operators.csv', '../exporter/paddledet', './paddledet_full_result.csv', warmup=1))
def get_param_of_all_detection_and_test(request):
    return request.param

def test_all_classify(get_param_of_all_classify_and_test):
    param = get_param_of_all_classify_and_test
    print('######Classify All Models Test######')
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

def test_all_segmentation(get_param_of_all_segmentation_and_test):
    param = get_param_of_all_segmentation_and_test
    print('######Segmentation All Models Test######')
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

def test_all_detection(get_param_of_all_detection_and_test):
    param = get_param_of_all_detection_and_test
    print('######Detection All Models Test######')
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

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddleclas_filtered_operators.csv', '../exporter/paddleclas', './paddleclas_filtered_result.csv', warmup=1))
def get_param_of_filtered_classify_and_test(request):
    return request.param

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddleseg_filtered_operators.csv', '../exporter/paddleseg', './paddleseg_filtered_result.csv', warmup=1))
def get_param_of_filtered_segmentation_and_test(request):
    return request.param

@pytest.fixture(params=create_params_list_by_csv_file('../analyzer/paddledet_filtered_operators.csv', '../exporter/paddledet', './paddledet_filtered_result.csv', warmup=1))
def get_param_of_filtered_detection_and_test(request):
    return request.param

def test_filtered_classify(get_param_of_filtered_classify_and_test):
    param = get_param_of_filtered_classify_and_test
    print('######Classify Filtered Models Test######')
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

def test_filtered_segmentation(get_param_of_filtered_segmentation_and_test):
    param = get_param_of_filtered_segmentation_and_test
    print('######Segmentation Filtered Models Test######')
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

def test_filtered_detection(get_param_of_filtered_detection_and_test):
    param = get_param_of_filtered_detection_and_test
    print('######Detection Filtered Models Test######')
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


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--test_mode", type=str, default='filtered', choices=['all', 'filtered', 'all_classify', 'all_detection', 'all_segmentation', 'filtered_classify', 'filtered_detection', 'filtered_segmentation'], help="test mode, all: execute all models, filtered: only execute the filtered models, all_classify: execute all classify models, all_detection: execute all detection models, all_segmentation: execute all segmentation models, filtered_classify: only execute filtered classify models, filtered_segmentation: only execute filtered segmentation models, filtered_detection: only execute filtered detection models")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.test_mode == 'all':
        # pytest.main(['-s','pytest_executor.py'])
        pytest.main(['pytest_executor.py::test_all_classify', 'pytest_executor.py::test_all_segmentation', 'pytest_executor.py::test_all_detection'])
    elif args.test_mode == 'filtered':
        pytest.main(['pytest_executor.py::test_filtered_classify', 'pytest_executor.py::test_filtered_segmentation', 'pytest_executor.py::test_filtered_detection'])
    elif args.test_mode == 'all_classify':
        pytest.main(['-v', 'pytest_executor.py::test_all_classify'])
    elif args.test_mode == 'all_detection':
        pytest.main(['-v', 'pytest_executor.py::test_all_detection'])
    elif args.test_mode == 'all_segmentation':
        pytest.main(['-v', 'pytest_executor.py::test_all_segmentation'])
    elif args.test_mode == 'filtered_classify':
        pytest.main(['-v', 'pytest_executor.py::test_filtered_classify'])
    elif args.test_mode == 'filtered_detection':
        pytest.main(['-v', 'pytest_executor.py::test_filtered_detection'])
    elif args.test_mode == 'filtered_segmentation':
        pytest.main(['-v', 'pytest_executor.py::test_filtered_segmentation'])
