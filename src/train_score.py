import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import argparse
import cPickle as pickle
import json
import numpy as np
import os
import pandas as pd
import seaborn as sns

## suppress warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

## mode ID is only command-line argument
## store as variable model_id and assert
## that path models/model_id exists,
## then change to that directory
parser = argparse.ArgumentParser()
parser.add_argument('model_id', help='model ID/directory name')
args = parser.parse_args()
model_id = args.model_id 
assert os.path.exists('models/{}'.format(model_id))
os.chdir('models/{}'.format(model_id))

## load model configuration file
model_dict = json.load(open('model.json','r'))

def get_model_obj(model_dict):
    '''using the string version of model
    e.g. xgboost.XGBClassifier, load that 
    object leveraging importlib library'''
    import importlib

    model_class_str = model_dict['model']
    model_obj_path = '.'.join(model_class_str.split('.')[:-1])
    model_name = model_class_str.split('.')[-1]
    model_package = importlib.import_module(model_obj_path)
    model_class = getattr(model_package, model_name)
    in_memory = model_class_str.split('.')[0] in ['sklearn','xgboost']
    
    return (model_class, in_memory)

def cv_train(model_dict, training, scoring_only, model_obj):
    '''given training/scoring data, model dict, 
    and a model obj, train k models plus the entire
    dataset and return dict of (set name, fitted model) pairs'''
    training_df = training.set_index(model_dict['index'])
    scoring_only_df = scoring_only.set_index(model_dict['index'])
    
    ## folds
    if model_dict['kfolds'] > 1:
        training_scoring_dict  = {
            f: {'train': training_df[training_df['fold'] != f],
                'score': training_df[training_df['fold'] == f]}
            for f in np.arange(model_dict['kfolds'])
        }
    else:
        training_scoring_dict = {}
    ## full sets
    training_scoring_dict['full'] = {
        'train': training_df,
        'score': scoring_only_df
    }
    
    feats = sorted(model_dict['features_list'])
    for set_nbr, set_data in training_scoring_dict.iteritems():
        if set_data['train'].shape[0] > 0:
            training_scoring_dict[set_nbr]['model'] = \
                model_obj(
                    **model_dict['model_params']
                ).fit(
                    np.array(set_data['train'][feats].values.tolist()),
                    set_data['train']['label'].ravel()
                )
    return training_scoring_dict
    
def cv_score(model_dict, training_scoring_dict):
    '''takes the model dict and another dict:
    keys are datasets (fold number or "full"),
    values dict with values of
    (training_df, scoring_only, fitted model obj).
    return pandas DF of scores'''
    
    feats = sorted(model_dict['features_list'])
    for i, (set_nbr, mdl) in enumerate(training_scoring_dict.iteritems()):
        curr_scoring = mdl['score']
        if curr_scoring.shape[0] > 0:
            curr_scoring['score'] = mdl['model'].predict_proba(
                                        curr_scoring[feats].values.tolist()
                                    )[:,1]
            if i == 0:
                scores_df = curr_scoring
            else:
                scores_df = scores_df.append(curr_scoring)
    
    return scores_df

def score_holdout_set(model_dict, training_scoring_dict, holdout_df):
    '''using an already trained model, score the holdout set 
    and return the scores'''
    feats = sorted(model_dict['features_list'])
    fitted_model = training_scoring_dict['full']['model']
    holdout_df['score'] = fitted_model.predict_proba(
                                    holdout_df[feats].values.tolist()
                                )[:,1]
    return holdout_df

## START EXECUTION

## since the dataset is small, it's faster to load the training/scoring 
## from disk than to spin up a sparkcontext and load from tables.
## if there were more data, writing to hive table would be 
## better option.
training = pd.read_csv('cv_data/training.csv')
scoring_only = pd.read_csv('cv_data/scoring_only.csv')
    
if not os.path.exists('serialized_models'): 
    os.mkdir('serialized_models')
if not os.path.exists('scores'): 
    os.mkdir('scores')

## as opposed to spark:
model_obj, train_in_memory = get_model_obj(model_dict)
## if train_in_memory is False --> spark
## in memory is faster when possible
## spark can train/score
if train_in_memory is True:
    training_scoring_dict = cv_train(model_dict, training, 
                                     scoring_only, model_obj)

    ## save models
    [t['model']._Booster.save_model('serialized_models/model_{}.xgb'.format(k)) 
    for (k, t) in training_scoring_dict.iteritems()
    if 'model' in t.keys()]

    scores_df = cv_score(model_dict, training_scoring_dict)
    scores_df.to_csv('scores/reported_scores.csv')
    
    if model_dict['holdout_set']['score_using_full_model']:
        holdout = pd.read_csv('cv_data/holdout.csv')
        holdout = score_holdout_set(model_dict, training_scoring_dict, 
                                    holdout)
        holdout.to_csv('scores/holdout_scores.csv')

        print 'successfully completed training/scoring.'
else:
    print 'Spark ML model training/scoring not supported yet. Exiting.'
    sys.exit(1)