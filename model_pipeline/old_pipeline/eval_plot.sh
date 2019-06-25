#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

MODEL_ID=$1
MODEL_DIR=models/$1

python model_pipeline/eval_plot.py $MODEL_ID >> $MODEL_DIR/logs/out 2>> $MODEL_DIR/logs/out
