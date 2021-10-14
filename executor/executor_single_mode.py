import argparse
import os

from executor import OpenvinoExecutor, PaddlenlpPredictor, PaddleExecutor, PaddlePredictorExecutor, compare

def performance_and_accuracy_test_single_mode(model_file, batch_size=1, warmup=0, repeats=1, predictor_type: str='normal', openvino_api_type: str = 'sync'):
    try:
        paddle_warmup_frame_time = None
        paddle_repeats_per_frame_time = None
        openvino_warmup_frame_time = None
        openvino_repeats_per_frame_time = None
        status = 0
        if not os.path.exists(model_file):
            print('model file "{}" not exists. Please specify it with --model_file argument.'.format(model_file))
            return

        # get the pdiparams file path
        dir_name = os.path.dirname(model_file)
        config_base = os.path.basename(model_file)
        config_base = os.path.splitext(config_base)[0]
        model_params_file = dir_name + '/' + config_base + '.pdiparams'

        if not os.path.exists(model_params_file):
            print('model params file "{}" not exists. Please check it.'.format(model_params_file))
            return

        # create test data
        if predictor_type == 'nlp':
            pdpd_predict_executor = PaddlenlpPredictor(str(model_file), str(model_params_file))
            test_inputs = pdpd_predict_executor.generate_inputs(batch_size, language='English')
        elif predictor_type == 'normal':
            pdpd_executor = PaddleExecutor(model_file)
            test_inputs = pdpd_executor.generate_inputs(batch_size)
            pdpd_predict_executor = PaddlePredictorExecutor(str(model_file), str(model_params_file))
        else:
            print('have no this {} paddle predictor type. Please check it.'.format(predictor_type))

        ## paddle predictor inference
        pdpd_predict_executor.run(test_inputs, warmup, repeats)
        pdpd_predict_result = pdpd_predict_executor.get_inference_results()
        paddle_warmup_frame_time = "{:.2f}".format(pdpd_predict_executor.warmup_time) if pdpd_predict_executor.warmup_time is not None else 'None'
        paddle_repeats_per_frame_time = "{:.2f}".format(pdpd_predict_executor.repeat_time) if pdpd_predict_executor.repeat_time is not None else 'None'

        ## openvino inference
        ov_executor = OpenvinoExecutor(model_file, 10, openvino_api_type)
        ov_executor.run(test_inputs, warmup, repeats)
        ov_result = ov_executor.get_inference_results()
        openvino_warmup_frame_time = "{:.2f}".format(ov_executor.warmup_time) if ov_executor.warmup_time is not None else 'None'
        openvino_repeats_per_frame_time = "{:.2f}".format(ov_executor.repeat_time) if ov_executor.repeat_time is not None else 'None'
        status = 2

        # compare
        ov_result = [ov_result[k] for k in sorted(ov_result)]
        pdpd_predict_result = [pdpd_predict_result[k] for k in sorted(pdpd_predict_result)]
        res = compare(ov_result, pdpd_predict_result)

        if res:
            status = 3
        else:
            status = 4

    finally:
        print('[result:] model_file: {}'.format(model_file))
        if status == 0:
            print('[result:] pdpd_predict_executor inference failed.')

        if status == 1:
            print('[result:] pdpd_predict_executor inference success.')
            print('[result:] ov_executor inference failed.')

        if status == 2:
            print('[result:] pdpd_predict_executor inference success.')
            print('[result:] ov_executor inference success.')
            print('[result:] compare failed.')

        if status == 3:
            print('[result:] pdpd_predict_executor inference success.')
            print('[result:] ov_executor inference success.')
            print('[result:] compare result equal.')

        if status == 4:
            print('[result:] pdpd_predict_executor inference success.')
            print('[result:] ov_executor inference success.')
            print('[result:] compare result not equal.')

        print('[result:] paddle warmup average time: {}(ms)'.format(paddle_warmup_frame_time))
        print('[result:] paddle repeates average time: {}(ms)'.format(paddle_repeats_per_frame_time))
        print('[result:] openvino warmup average time: {}(ms)'.format(openvino_warmup_frame_time))
        print('[result:] openvino repeates average time: {}(ms)'.format(openvino_repeats_per_frame_time))


def main():
    args = parse_args()
    performance_and_accuracy_test_single_mode(args.model_file, args.batch_size, args.warmup, args.repeats, args.paddle_predictor_type, args.openvino_api_type)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_file", type=str, default='', help="model filename, default(None)")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size, default(1)")
    parser.add_argument("--warmup", type=int, default=0, help="warm up inference, default(0)")
    parser.add_argument("--repeats", type=int, default=1, help="repeat number of inference, default(1)")
    parser.add_argument("--openvino_api_type", type=str, default='sync', choices=['sync', 'async'], help="select openvino inference api type, default(sync)")
    parser.add_argument("--paddle_predictor_type", type=str, default='normal', choices=['normal', 'nlp'], help="select paddle predictor type, default(normal)")
    return parser.parse_args()

if __name__ == "__main__":
    main()


# usage:
# NLP
# python3 executor_single_mode.py --model_file=../exporter/paddlenlp/bert-base-cased/model.pdmodel --paddle_predictor_type=nlp --batch_size=1 --warmup=1 --repeats=10