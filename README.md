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

## exporter example of Paddle OMZ
1. PaddleClas
``` python3 tools/export_model.py     -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml     -o Global.pretrained_model=../PaddleOMZAnalyzer/downloader/paddleclas/MobileNetV3_large_x1_0_pretrained -o Global.device=cpu -o Global.save_inference_dir=../PaddleOMZAnalyzer/exporter/paddleclas/ ```

2. PaddleDetection
``` python tools/export_model.py     -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml     -o use_gpu=false weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams     TestReader.inputs_def.image_shape=[3,608,608]     --output_dir ../PaddleOMZAnalyzer/exporter/paddledetection/ ```





