#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

MODEL_ID=$1
MODEL_DIR=models/$1

source model_pipeline/spark_submit_init.sh
spark-submit \
   --conf 'spark.executor.extraJavaOptions=-Dlog4j.configuration=/spark_logs/executor' \
   --conf 'spark.driver.extraJavaOptions=-Dlog4j.configuration=/spark_logs/driver' \
   model_pipeline/check_json_files.py $MODEL_ID > $MODEL_DIR/logs/out 2> $MODEL_DIR/logs/err
