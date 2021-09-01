#!/bin/bash

# https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst

# preprequisites
# pip install --upgrade paddlenlp>=2.0.0rc -i https://pypi.org/simple

paddlenlp_dir=$PWD/../../PaddleNLP/
omzanlayzer_dir=$PWD/../

# https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#%E9%A2%84%E6%B5%8B
bert_list="
            bert-base-uncased
            bert-large-uncased
            bert-base-cased
            bert-large-cased
            bert-base-multilingual-uncased
            bert-base-multilingual-cased
            bert-base-chinese
            bert-wwm-chinese
            bert-wwm-ext-chinese
            simbert-base-chinese
            "
for BERTVERSION in $bert_list;do
    echo $BERTVERSION
    BERT=$omzanlayzer_dir/exporter/paddlenlp/$BERTVERSION
    if [ ! -d "$BERT" ]; then
        echo "INFO: exporting $BERT"
        cd $paddlenlp_dir/examples/language_model/bert
        python3 export_model.py     --model_type bert     --model_path $BERTVERSION     --output_path $BERT/model
    fi
done

# https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer#%E5%AF%BC%E5%87%BA%E9%9D%99%E6%80%81%E5%9B%BE%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B%E4%B8%8E%E9%A2%84%E6%B5%8B%E5%BC%95%E6%93%8E%E9%A2%84%E6%B5%8B
# Modify yaml/inference_model_dir first - TODO
TRANSFORMER=$omzanlayzer_dir/exporter/paddlenlp/transformer/
if [ ! -d "$TRANSFORMER" ]; then
    echo "INFO: exporting $TRANSFORMER"
    cd $paddlenlp_dir/examples/machine_translation/transformer
    # python3 export_model.py --config ./configs/transformer.base.yaml
fi

# https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/waybill_ie
# python download.py --data_dir ./waybill_ie
WAYBILL_IE=$omzanlayzer_dir/exporter/paddlenlp/waybill_ie
if [ ! -d "$WAYBILL_IE" ]; then
    echo "INFO: exporting $WAYBILL_IE"
    cd $paddlenlp_dir/examples/information_extraction/waybill_ie
    python3 export_model.py --params_path ernie_ckpt/model_80.pdparams --output_path=$WAYBILL_IE
fi