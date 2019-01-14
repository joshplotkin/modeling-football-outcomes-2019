#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

ls models/$1
mkdir -p models/$1/logs
rm -f models/$1/{out,err,*png,*csv,*xgb} models/$1/*/{out,err,*png,*csv,*xgb}

time sh src/check_model_json.sh $1 && \
time sh src/cv_data.sh $1 && \
time sh src/train_score.sh $1 && \
time sh src/eval_plot.sh $1
