TRIAL_NAME=id
MODEL_CONFIG=config.json
MODELS_DIR=/Users/joshplotkin/Dropbox/data_science/modeling-football-outcomes/models
SOURCE_MODEL=0320_with_rankings_winner_20feats_noml

OUT_DIR=$MODELS_DIR/$TRIAL_NAME
mkdir $OUT_DIR
echo $OUT_DIR 

python hyperparam_tuning_bayesian_optimization.py $TRIAL_NAME $MODEL_CONFIG $MODELS_DIR $SOURCE_MODEL #> $OUT_DIR/out 2> $OUT_DIR/err
