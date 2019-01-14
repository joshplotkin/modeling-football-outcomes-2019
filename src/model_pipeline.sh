#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

if [ ! -d "models/$1" ]; then
  echo "directory models/$1 does not exist. exiting..."
  exit
fi

mkdir -p models/$1/logs
rm -f models/$1/{out,err,*png,*csv,*xgb} models/$1/*/{out,err,*png,*csv,*xgb}

echo "Check JSON files" && \
time sh src/check_json_files.sh $1 && \
echo "" && \
echo "Cross-validation data" && \
time sh src/cv_data.sh $1 && \
echo "" && \
echo "Train and score" && \
time sh src/train_score.sh $1 && \
echo "" && \
echo "Evaluate and plot" && \
time sh src/eval_plot.sh $1