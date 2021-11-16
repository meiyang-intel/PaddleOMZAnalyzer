#!/bin/bash

python3 paddleclas_analyzer.py --config_file=../downloader/paddleclas_full.csv --result_save_file=./paddleclas_full_operators.csv

python3 paddledetection_analyzer.py --config_file=../downloader/paddledet_full.csv --result_save_file=./paddledet_full_operators.csv

python3 paddleseg_analyzer.py --config_file=../downloader/paddleseg_full.csv --result_save_file=./paddleseg_full_operators.csv
