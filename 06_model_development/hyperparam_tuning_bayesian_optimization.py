import argparse
from bayes_opt import BayesianOptimization
from collections import OrderedDict
import json
import numpy as np
import os
import pandas as pd
import pickle
sfrom sklearn.metrics import roc_auc_score
import sys
from timeit import default_timer as timer
import xgboost

sys.path.append('../modeling-football-outcomes/config')
sys.path.append('/Users/joshplotkin/Dropbox/data_science/modeling-football-outcomes/model_pipeline')
from train_score_functions import get_model_obj, store_feature_importance, check_bad_values, cv_train, cv_score, score_holdout_set, store_models

def xgb_crossval(alpha_exp, colsample_bylevel, colsample_bynode, colsample_bytree, eta,
                 gamma_min_split_loss_exp, lambda_exp, max_delta_step, max_depth,
                 min_child_weight_exp, n_estimators, subsample):

    training = source_model['training']
    scoring_only = source_model['scoring_only']
    model_dict = source_model['config']

    model_dict['model_params'] = {
        ## strings
        'booster': config['booster'],
        'tree_method': config['tree_method'],

        ## base
        'n_estimators': int(n_estimators),
        'eta': eta,

        ## sampling
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'colsample_bynode': colsample_bynode,
        'subsample': subsample,

        ## tree splitting
        'max_depth': int(max_depth),
        'gamma': 10 ** gamma_min_split_loss_exp,
        'min_child_weight': 10 ** min_child_weight_exp,
        'max_delta_step': max_delta_step,

        ## regularization
        'lambda': 10 ** lambda_exp,
        'alpha': 10 ** alpha_exp,
    }

    model_obj = xgboost.sklearn.XGBClassifier
    training_scoring_dict = cv_train(model_dict, training,
                                     scoring_only, model_obj)
    scores_df = cv_score(model_dict, training_scoring_dict)
    score_pred = scores_df[['label', 'score']] \
        .reset_index(drop=False) \
        .set_index('game_id')

    return roc_auc_score(score_pred['label'], score_pred['score'])

def get_dir_id(dirs):
    def is_int(d):
        try:
            int(d)
            return True
        except:
            return False

    min_id = min(set(np.arange(100000)) - set(map(int, (filter(is_int, dirs)))))
    return '{:06d}'.format(min_id)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_name', help='trial name (id for incremental #)')
    parser.add_argument('model_config', help='location of model config JSON file')
    parser.add_argument('models_dir', help='models base directory--relative path ok')
    parser.add_argument('source_model', help='source model from which to get data')

    return parser.parse_args()

## TODO: move this into shell script
def setup_trial_dir(args, config):
    if args.trial_name == 'id':
        trial_name = get_dir_id(os.listdir(args.models_dir))
    else:
        trial_name = args.trial_name

    trial_dir = os.path.join(args.models_dir, trial_name)
    os.mkdir(trial_dir)
    os.chdir(trial_dir)
    json.dump(config, open('config.json', 'w'))

    return config

def load_source_model(args):
    source = os.path.join(args.models_dir, args.source_model)
    model_dict = json.load(open(os.path.join(source, 'model.json')))
    training = pd.read_csv(os.path.join(source, 'cv_data/training.csv'))
    scoring_only = pd.read_csv(os.path.join(source, 'cv_data/scoring_only.csv'))
    return {'config': model_dict,
            'training': training,
            'scoring_only': scoring_only}

def separate_fixed_variable_params(config):
    variable_params = {k: v for k, v in config.items() if type(v) is list}
    fixed_params = {k: v for k, v in config.items() if type(v) is not list}

    # need ordered variable params so it lines up with function call
    variable_params_ordered = OrderedDict()
    for ksort in sorted(variable_params.keys()):
        variable_params_ordered[ksort] = variable_params[ksort]

    return fixed_params, variable_params_ordered

args = parse_args()

config = json.load(open(args.model_config))

setup_trial_dir(args, config)
source_model = load_source_model(args)
fixed_params, variable_params = separate_fixed_variable_params(config)

## how to pass in *args?
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def call_optimizer(variable_params):
    return BayesianOptimization(
        f=xgb_crossval,
        pbounds=variable_params,
        verbose = 99,
    )

optimizer = call_optimizer(variable_params)

start = timer()
optimizer.maximize(
    init_points=config['init_points'],
    n_iter=config['n_iter'],
    # What follows are GP regressor parameters
    alpha=config['alpha'],
    n_restarts_optimizer=config['n_restarts_optimizer']
)
print('Time: ', timer() - start)

print("Final result:", optimizer.max)

with open('optimizer.pkl','wb') as w:
    pickle.dump(optimizer, w)

optimizer.max['target'].res
optimizer.res

## run the optimal model when done!

## pair plot of hyper-parameters

## eventually add features into the config... set up data first if new config?