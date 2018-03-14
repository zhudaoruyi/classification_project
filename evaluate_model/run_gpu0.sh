#/usr/bin/env bash
CODEPATH=/home/abc/pzw
#PYTHONPATH=/usr/local/lib
PYTHONPATH=${CODAPATH}
CUDA_VISIBLE_DEVICES=0 python2 ${CODEPATH}/evaluate_model/$1

