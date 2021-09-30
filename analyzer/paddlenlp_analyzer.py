import os
import sys
import subprocess
import re
import csv

from parser import parse_model_ops

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))

    with open("paddlenlp_filtered_operators.csv", 'w', newline='') as csvfile:
        # title for each column
        writer = csv.writer(csvfile, delimiter=',')
        # writer.writerow(['model', 'model_config', 'model_params', 'operator_set', 'unsupported_ops'])

        #BERT
        language_models = ['bert-base-uncased', 
                    'bert-large-uncased', 
                    'bert-base-cased',
                    'bert-large-cased',
                    'bert-base-multilingual-uncased',
                    'bert-base-multilingual-cased',
                    'bert-base-chinese',
                    'bert-wwm-chinese',
                    'bert-wwm-ext-chinese',
                    'simbert-base-chinese'
                    ]

        language_models += ['ernie-1.0',
                    'ernie-tiny',
                    'ernie-2.0-en',
                    'ernie-2.0-en-finetuned-squad',
                    'ernie-2.0-large-en',
                    'ernie-doc-base-zh',
                    'ernie-doc-base-en',
                    'ernie-gen-base-en',
                    'ernie-gen-large-en',
                    'ernie-gen-large-en-430g',
                    'ernie-gram-zh'
                    ]

        for lm in language_models:
            test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/{}'.format(lm)))
            if os.path.exists(test_model):
                operator_set, unsupported_ops = parse_model_ops(test_model)
                writer.writerow([lm, '', '', ','.join(operator_set), ','.join(unsupported_ops) if len(unsupported_ops)>0 else 'None'])

        # waybill_ie
        test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/waybill_ie'))
        operator_set, unsupported_ops = parse_model_ops(test_model)
        writer.writerow(['waybill_ie', '', '', ','.join(operator_set), ','.join(unsupported_ops) if len(unsupported_ops)>0 else 'None'])

        # transformer
        # base
        test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/transformer/base'))
        operator_set, unsupported_ops = parse_model_ops(test_model)
        writer.writerow(['transformer/base', '', '', ','.join(operator_set), ','.join(unsupported_ops) if len(unsupported_ops)>0 else 'None'])

        # big
        test_model = os.path.abspath(os.path.join(__dir__, '../exporter/paddlenlp/transformer/big'))
        operator_set, unsupported_ops = parse_model_ops(test_model)
        writer.writerow(['transformer/big', '', '', ','.join(operator_set), ','.join(unsupported_ops) if len(unsupported_ops)>0 else 'None'])
