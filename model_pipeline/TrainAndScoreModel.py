import importlib
import joblib
import numpy as np
import os

class TrainAndScoreModel:
    def __init__(self, model_dict, is_classification):
        self.model_dict = model_dict
        self.is_classification = is_classification
        self.save_models = model_dict['save']['serialized_models']
        self.save_cv_scores = model_dict['save']['cv_scores']
        self.save_holdout_scores = (model_dict['save']['holdout_scores']) \
                                    & (model_dict['actions']['do_score_holdout'])

        if self.save_models:
            save_loc = os.path.join(model_dict['models_dir'],
                                    model_dict['model_id'])
            self.models_dir = '{}/serialized_models'.format(save_loc)
            if not os.path.exists(self.models_dir):
                os.mkdir(self.models_dir)

        if (self.save_cv_scores) | (self.save_holdout_scores):
            save_loc = os.path.join(model_dict['models_dir'],
                                    model_dict['model_id'])
            self.scores_dir = '{}/scores'.format(save_loc)
            if not os.path.exists(self.scores_dir):
                os.mkdir(self.scores_dir)

    def get_model_object(self):
        """using the string version of model
        e.g. xgboost.XGBClassifier, load that
        object leveraging importlib library"""
        model_class_str = self.model_dict['model']
        model_obj_path = '.'.join(model_class_str.split('.')[:-1])
        model_name = model_class_str.split('.')[-1]
        model_package = importlib.import_module(model_obj_path)
        model_class = getattr(model_package, model_name)

        if model_obj_path == 'xgboost':
            # XGBoost errors out if this isn't set. However, it comes
            # with a warning message.
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        return model_class(**self.model_dict['model_params'])

    def store_models(self, model, set_nbr):
        library = self.model_dict['model'].split('.')[0]
        if library == 'xgboost':
            model._Booster.save_model(
               '{}/model_{}.xgb'.format(self.models_dir, set_nbr)
            )
        elif library == 'sklearn':
            joblib.dump(model, '{}/model_{}.xgb'.format(self.models_dir, set_nbr))
        else:
            print('currently only support sklearn and xgboost')
            sys.exit(1)

    def cv_train(self, training, scoring_only):
        """given training/scoring data, model dict,
        and a model obj, train k models plus the entire
        dataset and return dict of (set name, fitted model) pairs"""
        training_df = training.set_index(self.model_dict['index'])
        scoring_only_df = scoring_only.set_index(self.model_dict['index'])

        ## folds
        if self.model_dict['kfolds'] > 1:
            training_scoring_dict = {
                f: {'train': training_df[training_df['fold'] != f],
                    'score': training_df[training_df['fold'] == f]}
                for f in np.arange(self.model_dict['kfolds'])
            }
        else:
            training_scoring_dict = {}
        ## full sets
        training_scoring_dict['full'] = {
            'train': training_df,
            'score': scoring_only_df
        }

        for set_nbr, set_data in training_scoring_dict.items():
            if set_data['train'].shape[0] > 0:
                model = self.get_model_object()
                model.fit(
                    set_data['train'][self.model_dict['features_list']],
                    set_data['train']['label'].ravel()
                )
                training_scoring_dict[set_nbr]['model'] = model

                if self.save_models is True:
                    self.store_models(model, set_nbr)

        return training_scoring_dict

    def cv_train_and_score(self, training, scoring_only):
        """takes the model dict and another dict:
        keys are datasets (fold number or "full"),
        values dict with values of
        (training_df, scoring_only, fitted model obj).
        return pandas DF of scores"""
        training_scoring_dict = self.cv_train(training, scoring_only)
        score_col = {True: 'score',
                      False: 'regression_score'}\
                    [self.is_classification]

        for i, (set_nbr, mdl) in enumerate(training_scoring_dict.items()):
            curr_scoring = mdl['score']
            if curr_scoring.shape[0] > 0:
                if self.is_classification is True:
                    curr_scoring.loc[:, score_col] = mdl['model'].predict_proba(
                        curr_scoring[self.model_dict['features_list']]
                    )[:, 1]
                else:
                    curr_scoring.loc[:, score_col] = mdl['model'].predict(
                        curr_scoring[self.model_dict['features_list']]
                    )

                if i == 0:
                    cv_scores = curr_scoring
                else:
                    cv_scores = cv_scores.append(curr_scoring)

        if not self.is_classification:
            cv_scores = cv_scores.rename(columns={'label': 'regression_label'})
            cv_scores['regression_label'] = cv_scores['regression_label'].astype(float)

        self.cv_scores = cv_scores
        if self.save_cv_scores:
            self.cv_scores.to_csv('{}/cv_scores.csv'.format(self.scores_dir))

        self.model_objects = {set_nbr : mdl['model'] for set_nbr, model
                            in training_scoring_dict.items()}

    def score_holdout(self, model, holdout):
        score_col = {True: 'score',
                      False: 'regression_score'}\
                    [self.is_classification]
        if holdout.shape[0] > 0:
            if self.is_classification is True:
                holdout.loc[:, score_col] = model.predict_proba(
                    holdout[self.model_dict['features_list']]
                )[:, 1]
            else:
                holdout.loc[:, score_col] = model.predict(
                    holdout[self.model_dict['features_list']]
                )
        if self.save_holdout_scores:
            holdout.to_csv('{}/holdout_scores.csv'.format(self.scores_dir))
        return holdout
