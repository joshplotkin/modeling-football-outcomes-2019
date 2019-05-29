import argparse
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_dir', help='trial name (id for incremental #)')
    parser.add_argument('source_model', help='source model from which to get data')

    return parser.parse_args()


def get_optimizer(config, features):
    return Optimizer(
        [Categorical(categories=[0, 1], name=f, prior=[0.75, 0.25]) for f in features],
        #[Integer(low=0, high=1, name=f) for f in features],
        base_estimator=config.get('base_estimator', 'ET'),
        acq_optimizer=config.get('acq_optimizer', 'sampling'),
        n_random_starts=config.get('n_random_starts', None),
        n_initial_points=config.get('n_initial_points', 10),
        acq_func=config.get('acq_func', 'gp_hedge'),
        random_state=config.get('random_state', None),
        acq_func_kwargs=config.get('acq_func_kwargs', None),
        acq_optimizer_kwargs=config.get('acq_optimizer_kwargs', None)
    )


def load_source_model(args):
    model_dict = json.load(open(os.path.join(args.source_model, 'model.json')))
    training = pd.read_csv(os.path.join(args.source_model, 'cv_data/training.csv'))
    scoring_only = pd.read_csv(os.path.join(args.source_model, 'cv_data/scoring_only.csv'))
    return {'config': model_dict,
            'training': training,
            'scoring_only': scoring_only}


def train_score_evaluate(source_model):
    training = source_model['training']
    scoring_only = source_model['scoring_only']
    model_dict = source_model['config']

    model_dict['model_params'] = {
        ## strings
        'booster': 'gblinear',
        ## regularization
        'lambda': 21.300838,
        'alpha': 0.176392,
    }

    model_obj = xgboost.sklearn.XGBClassifier
    training_scoring_dict = cv_train(model_dict, training,
                                     scoring_only, model_obj)
    scores_df = cv_score(model_dict, training_scoring_dict)
    score_pred = scores_df[['label', 'score']] \
        .reset_index(drop=False) \
        .set_index('game_id')

    return roc_auc_score(score_pred['label'], score_pred['score'])


args = parse_args()
os.chdir(args.trial_dir)
config = json.load(open('config.json'))
source_model = load_source_model(args)

all_features = source_model['config']['features_list']
optimizer = get_optimizer(config, all_features)

## TEMP
# import pickle
# optimizer = pickle.load(open('/Users/joshplotkin/Dropbox/data_science/modeling-football-outcomes/models/000014/optimizer.pkl','rb'))
iter = 0
## END TEMP

header = all_features + ['auc']

#TODO: add in ability to lock in/exclude features
# fixed_in_features = []
# fixed_out_features = []

history = {}
while True:
    binary_vector = optimizer.ask()

    assert len(binary_vector) == len(all_features)
    source_model['config']['features_list'] = [f for i, f in zip(binary_vector, all_features) if i == 1]

    run_auc = train_score_evaluate(source_model)

    history[iter] = dict(zip(header, binary_vector + [run_auc]))
    optimizer.tell(binary_vector, -run_auc)

    iter += 1

    save_interval = 10
    if (iter > 0) & (iter % save_interval == 0):
        # renaming arobics because saving takes awhile
        # and killing process in the midst
        with open('optimizer_new.pkl','wb') as w:
            pickle.dump(optimizer, w)

        if os.path.exists('optimizer.pkl'):
            os.remove('optimizer.pkl')
        os.rename('optimizer_new.pkl','optimizer.pkl')
        pd.DataFrame.from_dict(history, orient='index').to_csv('results.csv')
        # print(pd.DataFrame.from_dict(history, orient='index').tail(show_interval))

## note: left out dart because it's slow

## run the optimal model when done!

## pair plot of hyper-parameters

## eventually add features into the config... set up data first if new config?

## things to do next
# make it flexible enough to take model type and things like xgboost booster type
# combine hyperparameter tuning and feature selection
# do some bagging to reduce variance in finding "best" model. trials are fast, so maybe execute the full pipeline 5-10 times and average
# rank correlations / kendall-tau distance
# make it flexible enough to take something other than AUC. AUC is likely not best anyway
# look at holdout set results on "best" model
# 'where do we miss?' analysis
# * week #
# * closer money lines?
# look into diff features, e.g. home rank minus visitor rank
# look into which prediction type is best for a specific goal, e.g. super contest style
# more features: home - visitor... offense vs defense
# more data (rankings?)
# does gblinear need feature interactions? idea could be pass thru kernel and feature select
# needs to be able to read off initial optimizer

# think metrics,,, any input/prediction type can be evaluated the same way. super-contest style: accuracy for top n predictions.
# o/u is probably