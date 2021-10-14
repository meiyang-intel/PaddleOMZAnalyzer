#!/bin/bash

# so far only wide_deep available

# download static model
wget https://paddlerec.bj.bcebos.com/wide_deep/wide_deep.tar

mkdir -p paddlerec/wide_deep
tar -xvf wide_deep.tar -C paddlerec/wide_deep

rm wide_deep.tar