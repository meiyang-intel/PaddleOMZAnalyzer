import pytest

from executor_single_mode import performance_and_accuracy_test_single_mode

# PaddleNLP
@pytest.mark.parametrize(
    "model_file",
    [
        '../exporter/paddlenlp/bert-base-cased/model.pdmodel',
        '../exporter/paddlenlp/bert-wwm-chinese/model.pdmodel',
        '../exporter/paddlenlp/ernie-1.0/model.pdmodel',
        '../exporter/paddlenlp/waybill_ie/inference.pdmodel',
        '../exporter/paddlenlp/transformer/base/transformer.pdmodel'
    ]
)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_paddlenlp(model_file, batch_size):
    performance_and_accuracy_test_single_mode(model_file, batch_size, 1, 2, 'nlp')


# PaddleOCR
@pytest.mark.parametrize(
    "model_file",
    [
        '../exporter/paddleocr/ch_ppocr_mobile_v2.0_det_infer/inference.pdmodel',
        '../exporter/paddleocr/ch_ppocr_mobile_v2.0_rec_infer/inference.pdmodel',
        '../exporter/paddleocr/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel',
        '../exporter/paddleocr/ch_PP-OCRv2_det_infer/inference.pdmodel',
        '../exporter/paddleocr/ch_PP-OCRv2_rec_infer/inference.pdmodel',
        '../exporter/paddleocr/det_mv3_db_v2.0/inference.pdmodel',
        '../exporter/paddleocr/rec_mv3_none_bilstm_ctc_v2.0/inference.pdmodel'
    ]
)
@pytest.mark.parametrize("batch_size", [1])
def test_paddleocr(model_file, batch_size):
    performance_and_accuracy_test_single_mode(model_file, batch_size, 1, 2, 'normal')


# PaddleRec
@pytest.mark.parametrize(
    "model_file",
    [
        '../exporter/paddlerec/wide_deep/rec_inference.pdmodel'
    ]
)
@pytest.mark.parametrize("batch_size", [1])
def test_paddlerec(model_file, batch_size):
    performance_and_accuracy_test_single_mode(model_file, batch_size, 1, 2, 'nlp')