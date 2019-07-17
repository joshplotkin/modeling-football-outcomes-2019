import json
import numpy as np
import os
import pandas as pd
import sys

from ExecuteModelPipeline import ExecuteModelPipeline

class Ensemble:
    def __init__(self, config):
        self.config=config
        self.trial_path=self.get_trial_path()

    def get_trial_path(self):
        return os.path.join(self.config['models_dir'], self.config['ensemble_model_id'])

    def setup_trial_dir(self, path, overwrite='Y'):
        overwrite = True if overwrite.upper()[0] == 'Y' else False
        if (overwrite is False) & (os.path.exists(path)):
            print('model path already exists and user input disallows overwriting. exiting...')
            sys.exit(1)
        if (overwrite) & (os.path.exists(path)):
            import shutil
            shutil.rmtree(path, ignore_errors=False)
            print(f'removed {path}...')
        os.mkdir(path)

    def get_model_id(self, model_nbr):
        return '{:05d}'.format(model_nbr)

    def get_model_path(self, model_nbr):
        return os.path.join(self.trial_path, self.get_model_id(model_nbr))

    def apply_parameters_for_iteration(self, model_dict, model_nbr):
        for param_to_change, param_values in self.config['input_changes_by_iteration'].items():
            model_dict[param_to_change] = param_values[model_nbr]
        return model_dict

    def apply_seeds(self, model_dict, model_nbr, seed):
        model_dict['dataset_seed'] = int(seed + model_nbr)
        model_dict['fold_seed'] = int(seed + model_nbr)
        return model_dict

    def modify_parameters_for_iteration(self, model_dicts, model_nbr, seed):
        model_dict = model_dicts[model_nbr]
        model_dict = self.apply_seeds(model_dict, model_nbr, seed)
        if 'input_changes_by_iteration' in self.config:
            return self.apply_parameters_for_iteration(model_dict, model_nbr)
        else:
            return model_dict

    def dump_evaluation_json_in_submodel(self, model_path, model_id):
        if not os.path.exists(self.config['evaluation_config']):
            print('Warning: evaluation.json file does not exist')
            return

        eval_dict = json.load(open(self.config['evaluation_config']))
        eval_dict['model_id'] = model_id
        eval_dict['ensemble_models'] = self.config['number_of_models']
        eval_dict['save']['plots'] = True

        json.dump(
            eval_dict,
            open(os.path.join(model_path, 'evaluation.json'), 'w'),
            indent=3
        )

    def create_ensemble_dir_structure(self):
        n_models = self.config['number_of_models']
        model_dicts = [json.load(open(self.config['source'])) for _ in np.arange(n_models)]
        seed = np.random.randint(1, 1000000)
        for model_nbr in np.arange(n_models):
            model_dict = self.modify_parameters_for_iteration(model_dicts, model_nbr, seed)
            model_path = self.get_model_path(model_nbr)
            self.setup_trial_dir(model_path)

            model_dict['model_id'] = '{}/{}'.format(self.config['ensemble_model_id'],
                                                    self.get_model_id(model_nbr))
            model_dict['models_dir'] = self.config['models_dir']
            json.dump(
                model_dict,
                open(os.path.join(model_path, 'model.json'), 'w'),
                indent=3
            )
            if self.config.get('evaluation_config', None):
                self.dump_evaluation_json_in_submodel(model_path, model_dict['model_id'])

    def execute_submodel(self, model_nbr):
        model_path = self.get_model_path(model_nbr)
        model_json_path = os.path.join(model_path, 'model.json')
        model_eval_path = os.path.join(model_path, 'evaluation.json')
        if os.path.exists(model_eval_path):
            ExecuteModelPipeline(model_json_path, model_eval_path, 'Y')
        else:
            ExecuteModelPipeline(model_json_path, None, 'Y')

    def train_and_score(self):
        for model_nbr in np.arange(self.config['number_of_models']):
            self.execute_submodel(model_nbr)

    def get_labels_tbl(self):
        if type(self.config['source']) is str:
            return json.load(open(self.config['source'],'r'))['labels_tbl']
        elif type(self.config['source']) is list:
            return json.load(open(self.config['source'][0],'r'))['labels_tbl']

    def get_labels_df(self):
        return pd.read_csv(
            'data/{}/{}.csv'.format(
                *self.get_labels_tbl().split('.')
            )
        )

    def scores_only_index_label_or_scores_cols(self, scores, model_dict):
        cols = [c for c in scores.columns if
                c in model_dict['index'] or c in ['label','score'] or c.endswith('_label') or c.endswith('_score')]
        return scores[cols]

    def scores_tbl_scores_cols_renamed(self, scores, model_nbr):
        score_rename = {c: '{}_{}'.format(model_nbr, c)
                        for c in scores.columns if c == 'score' or c.endswith('_score')}
        return scores.rename(columns=score_rename)

    def scores_tbl_labels_cols_renamed(self, scores, model_nbr):
        labels_rename = {c: '{}_{}'.format(model_nbr, c)
                         for c in scores.columns if c == 'label' or c.endswith('_label')}
        return scores.rename(columns=labels_rename)

    def scores_tbl_columns_renamed(self, scores, model_nbr):
        scores = self.scores_tbl_scores_cols_renamed(scores, model_nbr)
        return self.scores_tbl_labels_cols_renamed(scores, model_nbr)

    def load_and_prepare_scores(self, model_path, model_dict, model_nbr):
        scores = pd.read_csv(os.path.join(model_path, 'scores/cv_scores.csv'))
        scores = self.scores_only_index_label_or_scores_cols(scores, model_dict)
        return self.scores_tbl_columns_renamed(scores, model_nbr)

    def segment_labels_scores_cols(self, all_scores):
        labels_cols = [c for c in all_scores.columns if c == 'label' or c.endswith('_label')]
        scores_cols = [c for c in all_scores.columns if c == 'score' or c.endswith('_score')]
        label_col_base = '_'.join(labels_cols[0].split('_')[1:])
        score_col_base = '_'.join(scores_cols[0].split('_')[1:])

        return labels_cols, label_col_base, scores_cols, score_col_base

    def collapse_labels(self, all_scores, labels_cols, label_col_base):
        all_scores[label_col_base] = all_scores[labels_cols]\
                                            .apply(np.nanmean, axis=1)\
                                            .astype(float)
        all_scores['label'] = (all_scores[label_col_base] > 0).astype(int)
        return all_scores[~all_scores[label_col_base].isnull()]

    def generate_combined_scores(self, aggregation_method, all_scores_nonnull, labels_cols, scores_cols, score_col_base):
        agg_method = eval('np.nan{}'.format(aggregation_method))
        all_scores_nonnull[score_col_base] = all_scores_nonnull[scores_cols].apply(agg_method, axis=1)
        return all_scores_nonnull.drop(labels_cols, axis=1)

    def form_final_scores_table(self, aggregation_method, all_scores):
        labels_cols, label_col_base, scores_cols, score_col_base = self.segment_labels_scores_cols(all_scores)
        all_scores_nonnull = self.collapse_labels(all_scores, labels_cols, label_col_base)
        return self.generate_combined_scores(aggregation_method, all_scores_nonnull, labels_cols, scores_cols, score_col_base)

    def combine_scores(self, aggregation_method):
        all_scores = self.get_labels_df()
        for model_nbr in np.arange(self.config['number_of_models']):
            model_path = self.get_model_path(model_nbr)
            model_dict = json.load(open(os.path.join(model_path, 'model.json')))
            scores = self.load_and_prepare_scores(model_path, model_dict, model_nbr)
            all_scores = all_scores.merge(scores, on=model_dict['index'], how='left')

        all_scores = self.form_final_scores_table(aggregation_method, all_scores)
        return all_scores[~all_scores['score'].isnull()]
