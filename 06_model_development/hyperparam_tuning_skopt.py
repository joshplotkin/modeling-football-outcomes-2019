import argparse
from collections import OrderedDict
import json
import os
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
import sys
import xgboost

sys.path.append('../modeling-football-outcomes/config')
sys.path.append('/Users/joshplotkin/Dropbox/data_science/modeling-football-outcomes/model_pipeline')
from train_score_functions import get_model_obj, store_feature_importance, check_bad_values, cv_train, cv_score, score_holdout_set, store_models

# XGBoost errors out if this isn't set. However, it comes
# with a warning message.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train_score_evaluate(source_model, iteration_params):
    training = source_model['training']
    scoring_only = source_model['scoring_only']
    model_dict = source_model['config']

    model_dict['model_params'] = {
        ## strings
        'booster': iteration_params['booster'],
        'tree_method': iteration_params['tree_method'],

        ## base
        'n_estimators': iteration_params['n_estimators'],
        'eta': iteration_params['eta'],

        ## sampling
        'colsample_bytree': iteration_params['colsample_bytree'],
        'colsample_bylevel': iteration_params['colsample_bylevel'],
        'colsample_bynode': iteration_params['colsample_bynode'],
        'subsample': iteration_params['subsample'],

        ## tree splitting
        'max_depth': int(iteration_params['max_depth']),
        'gamma': iteration_params['gamma_min_split_loss'],
        'min_child_weight': iteration_params['min_child_weight'],
        'max_delta_step': iteration_params['max_delta_step'],

        ## regularization
        'lambda': iteration_params['lambda'],
        'alpha': iteration_params['alpha'],
    }

    # if model_dict['model_params']['booster'] == 'gblinear':
    #     model_dict['model_params']['feature_selector'] = 'greedy'

    model_obj = xgboost.sklearn.XGBClassifier
    training_scoring_dict = cv_train(model_dict, training,
                                     scoring_only, model_obj)
    scores_df = cv_score(model_dict, training_scoring_dict)
    score_pred = scores_df[['label', 'score']] \
        .reset_index(drop=False) \
        .set_index('game_id')

    return roc_auc_score(score_pred['label'], score_pred['score'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_dir', help='trial name (id for incremental #)')
    parser.add_argument('source_model', help='source model from which to get data')

    return parser.parse_args()

def load_source_model(args):
    model_dict = json.load(open(os.path.join(args.source_model, 'model.json')))
    training = pd.read_csv(os.path.join(args.source_model, 'cv_data/training.csv'))
    scoring_only = pd.read_csv(os.path.join(args.source_model, 'cv_data/scoring_only.csv'))
    return {'config': model_dict,
            'training': training,
            'scoring_only': scoring_only}

def separate_fixed_variable_params(config):
    variable_params = {k: v for k, v in config.items() if type(v) is dict}
    fixed_params = {k: v for k, v in config.items() if type(v) is not dict}

    # need ordered variable params so it lines up with function call
    variable_params_ordered_typed = OrderedDict()
    for k in sorted(variable_params.keys()):
        param = variable_params[k]
        val_range = param['values']
        if param['type'][:3].lower() == 'int':
            space_obj = Integer(low=val_range[0], high=val_range[1], name=k)

        elif param['type'][:4].lower() == 'real':
            space_obj = Real(low=val_range[0], high=val_range[1], name=k, prior=param.get('prior', 'uniform'))

        elif param['type'][:3].lower() == 'cat':
            default_prior = [1/len(val_range) for _ in val_range]
            space_obj = Categorical(categories=val_range, name=k, prior=param.get('prior', default_prior))

        else:
            print(f"Invalid type in config.json for key: {k}")
            sys.exit(0)

        variable_params_ordered_typed[k] = {'Space': space_obj}

    return fixed_params, variable_params_ordered_typed

def get_optimizer(config, variable_df):
    variable_df = variable_df.sort_index()
    return Optimizer(
        [v for v in variable_df['Space'].tolist()],
        base_estimator=config.get('base_estimator', 'ET'),
        acq_optimizer=config.get('acq_optimizer', 'sampling'),
        n_random_starts=config.get('n_random_starts', None),
        n_initial_points=config.get('n_initial_points', 10),
        acq_func=config.get('acq_func', 'gp_hedge'),
        random_state=config.get('random_state', None),
        acq_func_kwargs=config.get('acq_func_kwargs', None),
        acq_optimizer_kwargs=config.get('acq_optimizer_kwargs', None)
    )

args = parse_args()

os.chdir(args.trial_dir)
config = json.load(open('config.json'))

source_model = load_source_model(args)
fixed_params, variable_params = separate_fixed_variable_params(config)
variable_df = pd.DataFrame.from_dict(variable_params, orient='index')

optimizer = get_optimizer(config, variable_df)

iter = 0

header = variable_df.index.tolist() + list(fixed_params.keys()) + ['auc']
history = {}
while True:
    #print('pre-ask')
    variable_df['next'] = optimizer.ask()
    #print('post-ask')
    iteration_params = {**variable_df['next'].to_dict(), **fixed_params}
    #print(pd.DataFrame.from_dict(iteration_params, orient='index'))

    run_auc = train_score_evaluate(source_model, iteration_params)
    #print('post-score')

    history[iter] = dict(zip(header, variable_df['next'].tolist() + list(fixed_params.values()) + [run_auc]))
    optimizer.tell(variable_df['next'].tolist(), -run_auc)

    iter += 1

    show_interval = 10
    if (iter > 0) & (iter % show_interval == 0):
        with open('optimizer.pkl','wb') as w:
            pickle.dump(optimizer, w)
        pd.DataFrame.from_dict(history, orient='index').to_csv('results.csv')
        print(pd.DataFrame.from_dict(history, orient='index',).tail(show_interval))

## note: left out dart because it's slow

## run the optimal model when done!

## pair plot of hyper-parameters

## eventually add features into the config... set up data first if new config?