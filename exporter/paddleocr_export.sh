#!/bin/bash

# 1. PPOCR static models are downloadable
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/models_list.md

TARGET_DIR=$PWD/paddleocr
paddleocr_dir=$PWD/../../PaddleOCR/

mkdir -p $TARGET_DIR

model_list="https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar,\
https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar,\
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar,\
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar\
"

IFS=,
for model in $model_list;
do
    echo "Downloading $model ......"
    wget $model -O - | tar -xvf - -C $TARGET_DIR
done

# TODO: how to handle the exception case, which is gzip compressed?
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar -O - | tar -zxvf - -C $TARGET_DIR

# 2. 2-stage algorithms export
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/algorithm_overview.md

# DB
echo "INFO: Downloading det_mv3_db ......"
cd $paddleocr_dir
mkdir -p ./cache && wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar -O - | tar -xvf - -C ./cache
python3 tools/export_model.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model="./cache/det_mv3_db_v2.0_train/best_accuracy" Global.save_inference_dir="$TARGET_DIR/det_mv3_db_v2.0"

# rec_mv3_none_bilstm_ctc 
echo "INFO: Downloading rec_mv3_none_bilstm_ctc ......"
cd $paddleocr_dir
mkdir -p ./cache && wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar -O - | tar -xvf - -C ./cache
python3 tools/export_model.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml -o Global.pretrained_model="./cache/rec_mv3_none_bilstm_ctc_v2.0_train/best_accuracy" Global.save_inference_dir="$TARGET_DIR/rec_mv3_none_bilstm_ctc_v2.0"