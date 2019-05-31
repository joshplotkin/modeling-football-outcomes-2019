import importlib
import numpy as np
import os

class TrainAndScoreModel:
    def __init__(self, model_dict):
        self.model_dict = model_dict

    def init_model(self):
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

        self.model = model_class

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

        feats = sorted(self.model_dict['features_list'])
        for set_nbr, set_data in training_scoring_dict.items():
            if set_data['train'].shape[0] > 0:
                training_scoring_dict[set_nbr]['model'] = \
                    self.model(
                        **self.model_dict['model_params']
                    ).fit(
                        np.array(set_data['train'][feats].values.tolist()),
                        set_data['train']['label'].ravel()
                    )
        return training_scoring_dict

    def cv_score(self, training, scoring_only):
        """takes the model dict and another dict:
        keys are datasets (fold number or "full"),
        values dict with values of
        (training_df, scoring_only, fitted model obj).
        return pandas DF of scores"""
        training_scoring_dict = self.cv_train(training, scoring_only)
        feats = sorted(self.model_dict['features_list'])
        for i, (set_nbr, mdl) in enumerate(training_scoring_dict.items()):
            curr_scoring = mdl['score']
            if curr_scoring.shape[0] > 0:
                curr_scoring.loc[:, 'score'] = mdl['model'].predict_proba(
                    curr_scoring[feats].values.tolist()
                )[:, 1]
                if i == 0:
                    scores_df = curr_scoring
                else:
                    scores_df = scores_df.append(curr_scoring)

        self.cv_scores = scores_df