import csv
import os
import argparse
from glob import glob

from paddle_frontend import paddle_frontend_supported_ops
from paddle_parser import get_ops

# return all_ops and unsupported_ops in model.
def parse_model_ops(exported_path):
    operator_set = get_ops(exported_path)    

    # pick out unsupported operators
    unsupported_ops = []
    for op in operator_set:
        if op not in paddle_frontend_supported_ops:
            unsupported_ops.append(op)

    print(exported_path, operator_set, len(operator_set), unsupported_ops, len(unsupported_ops))

    return operator_set, unsupported_ops



def main(args):
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    model_zoo = os.path.join(__dir__, '../exporter', args.model_zoo)

    if not os.path.exists(model_zoo):
        print('model zoo {} not exists'.format(model_zoo))
        return

    with open("{}_filtered_operators.csv".format(args.model_zoo), 'w', newline='') as csvfile:
        # title for each column
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['model', 'model_config', 'model_params', 'operator_set', 'unsupported_ops'])

        paths = glob(model_zoo + "/*/", recursive = False)
        for model_path in paths:
            if os.path.isdir(model_path):
                operator_set, unsupported_ops = parse_model_ops(model_path)
                writer.writerow([os.path.dirname(model_path).split(os.path.sep)[-1], '', '', ','.join(operator_set), ','.join(unsupported_ops) if len(unsupported_ops)>0 else 'None'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_zoo", type=str, default='paddleocr', help="model zoo")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())