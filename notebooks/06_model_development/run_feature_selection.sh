#!/bin/bash

TRIAL_NAME=id
MODEL_CONFIG=config.json
MODELS_DIR=/Users/joshplotkin/Dropbox/data_science/modeling-football-outcomes/models
SOURCE_MODEL=0228_with_rankings_features_winner

TRIAL_NAME_MOD=`python initialize_trial_dir.py $TRIAL_NAME $MODEL_CONFIG $MODELS_DIR`
echo TRIAL: $TRIAL_NAME_MOD
OUT_DIR=$MODELS_DIR/$TRIAL_NAME_MOD
SOURCE_DIR=$MODELS_DIR/$SOURCE_MODEL

python feature_selection_skopt.py $OUT_DIR $SOURCE_DIR #> $OUT_DIR/out 2> $OUT_DIR/err
