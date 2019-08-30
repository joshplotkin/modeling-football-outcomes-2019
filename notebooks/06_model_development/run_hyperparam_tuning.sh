#!/bin/bash

TRIAL_NAME=id
MODEL_CONFIG=config.json
MODELS_DIR=/Users/joshplotkin/Dropbox/data_science/modeling-football-outcomes/models
SOURCE_MODEL=0320_with_rankings_winner_20feats_noml

TRIAL_NAME_MOD=`python initialize_trial_dir.py $TRIAL_NAME $MODEL_CONFIG $MODELS_DIR`
echo TRIAL: $TRIAL_NAME_MOD
OUT_DIR=$MODELS_DIR/$TRIAL_NAME_MOD
SOURCE_DIR=$MODELS_DIR/$SOURCE_MODEL

python hyperparam_tuning_skopt.py $OUT_DIR $SOURCE_DIR #> $OUT_DIR/out 2> $OUT_DIR/err
