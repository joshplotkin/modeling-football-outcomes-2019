#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

MODEL_ID=$1
MODEL_DIR=models/$1

sh src/spark_submit_init.sh
spark-submit \
   --conf 'spark.executor.extraJavaOptions=-Dlog4j.configuration=/spark_logs/executor' \
   --conf 'spark.driver.extraJavaOptions=-Dlog4j.configuration=/spark_logs/driver' \
   src/check_json_files.py $MODEL_ID > $MODEL_DIR/logs/out 2> $MODEL_DIR/logs/err
