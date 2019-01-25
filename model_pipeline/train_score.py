import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import argparse
import cPickle as pickle
from joblib import dump
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
    library = model_class_str.split('.')[0]
    
    return (model_class, library)

def store_feature_importance(model_dict, library, mdl, set_nbr):
    '''write feature importances for each set to csv'''
    feature_imp = {}
    feats = sorted(model_dict['features_list'])
    if library == 'xgboost':
        imp = mdl._Booster\
                 .get_score(fmap='', importance_type='gain')
        for i, (c, importance) in enumerate(imp.iteritems()):
            idx = int(c[1:])
            feature_imp[i] = {
                'Importance': importance,
                'Feature': feats[idx]
            }
    elif library == 'sklearn':
        imp_list = mdl.feature_importances_
        for i in range(len(feats)):
            feature_imp[i] = {'Importance': imp_list[i], 
                              'Feature': feats[i]}
    
    imp = pd.DataFrame.from_dict(
                feature_imp, orient='index'
            ).sort_values(
                by='Importance', ascending=False
            ).to_csv(
                'stats/reported/importance_{}.csv'
                   .format(set_nbr),
                index=False
            )

def check_bad_values(df):
    err_strs = []
    for c in df.columns:
        col_errs = []
        ## nulls
        if df[c].dropna().shape[0] != df[c].shape[0]:
            col_errs.append('null value(s)')
        if df[c].abs().max() == np.inf:
            col_errs.append('inf or -inf value(s)')
            
        if len(col_errs) == 0:
            pass
        elif len(col_errs) == 1:
            col_str = 'column {} has {}'\
                         .format(c, col_errs[0])
            err_strs.append(col_str)
        else:
            col_str = 'column {} has {} and {}'\
                         .format(c, *col_errs)
            err_strs.append(col_str)
        
    return err_strs 

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
            ## if library == 'xgboost' then warning
            ## if library == 'sklean' then fail
            err_strs = check_bad_values(set_data['train'][feats + ['label']])
            if (err_strs != []) & (library == 'sklearn'):
                print 'ERROR'
                print '\n'.join(err_strs)
                print 'Exiting...'
                sys.exit(1)
            elif (err_strs != []) & (library == 'xgboost'):
                print 'WARNING'
                print '\n'.join(err_strs)
            
            mdl = model_obj(
                    **model_dict['model_params']
                ).fit(
                    np.array(set_data['train'][feats].values.tolist()),
                    set_data['train']['label'].ravel()
                )

            store_feature_importance(model_dict, library, mdl, set_nbr)
            training_scoring_dict[set_nbr]['model'] = mdl

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

def store_models(library, training_scoring_dict):
    if library == 'xgboost':
        ## save models
        [t['model']._Booster.save_model(
                        'serialized_models/model_{}.xgb'.format(k)
                       ) 
        for (k, t) in training_scoring_dict.iteritems()
        if 'model' in t.keys()]
    elif library == 'sklearn':
        [dump(t['model'], 'serialized_models/model_{}.pkl'.format(k))
        for (k, t) in training_scoring_dict.iteritems()
        if 'model' in t.keys()]        
    else:
        print 'currently only support sklearn and xgboost'
        sys.exit(1)

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
if not os.path.exists('stats'): 
    os.mkdir('stats')
    os.mkdir('stats/reported')

## as opposed to spark:
model_obj, library = get_model_obj(model_dict)
## in memory is faster when possible
## spark can train/score on larger data when needed
if library in ['sklearn','xgboost']:
    training_scoring_dict = cv_train(model_dict, training, 
                                     scoring_only, model_obj)

    store_models(library, training_scoring_dict)

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