'''
arg parser
'''
import argparse

def str2bool(v):
    return v.lower() in ("true", "t", "1")

def init_args():
    parser = argparse.ArgumentParser()

    # params for text detector
    parser.add_argument("--homepage", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')  

    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

