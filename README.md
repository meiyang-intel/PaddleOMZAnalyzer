# PaddleOMZAnalyzer
An analyzer of paddlepaddle omz to for openvino.


## Prerequisite
1. install paddlepaddle>=2.1
2. git clone below model zoo repos. Make sure to checkout to corresponding branch of paddlepaddle, for example, "release/2.1". 
No need to compile them. Just put them at the same folder of PaddleOMZAnalyzer.

#### 1. PaddleClas
https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict 
#### 2. PaddleDetection
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md
#### 3. PaddleSeg
https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md 
#### 4. PaddleOCR 
It provides exported model, directly download inference_model.
https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15 
#### 5. PaddleNLP
[NLP pretrained model summary]
https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html#transformer
https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst

[BERT]
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#%E9%A2%84%E6%B5%8B

[Transformer]
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer#%E5%AF%BC%E5%87%BA%E9%9D%99%E6%80%81%E5%9B%BE%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B%E4%B8%8E%E9%A2%84%E6%B5%8B%E5%BC%95%E6%93%8E%E9%A2%84%E6%B5%8B

[BiGRU_CRF]
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/waybill_ie


## exporter example of Paddle OMZ
1. PaddleClas
``` python3 tools/export_model.py     -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml     -o Global.pretrained_model=../PaddleOMZAnalyzer/downloader/paddleclas/MobileNetV3_large_x1_0_pretrained -o Global.device=cpu -o Global.save_inference_dir=../PaddleOMZAnalyzer/exporter/paddleclas/ ```

2. PaddleDetection
``` python tools/export_model.py     -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml     -o use_gpu=false weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams     TestReader.inputs_def.image_shape=[3,608,608]     --output_dir ../PaddleOMZAnalyzer/exporter/paddledetection/ ```





