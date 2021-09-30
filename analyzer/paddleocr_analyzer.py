import csv
import os
import argparse
from glob import glob

from parser import parse_model_ops

def main(args):
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    model_zoo = os.path.join(__dir__, '../exporter', args.model_zoo)

    if not os.path.exists(model_zoo):
        print('model zoo {} not exists'.format(model_zoo))
        return

    with open("{}_filtered_operators.csv".format(args.model_zoo), 'w', newline='') as csvfile:
        # title for each column
        writer = csv.writer(csvfile, delimiter=',')
        # writer.writerow(['model', 'model_config', 'model_params', 'operator_set', 'unsupported_ops'])

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