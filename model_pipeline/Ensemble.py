import json
import numpy as np
import os
import pandas as pd
import sys

from ExecuteModelPipeline import ExecuteModelPipeline
from EvaluateModel import EvaluateAndPlot

# TODO: how to infer if it's classification?

class Ensemble:
    def __init__(self, model_path, eval_path):
        model_dict, eval_dict = self.load_json_files(model_path, eval_path)
        eval_dict['model_id'] = model_dict['ensemble_model_id']
        self.config=model_dict
        self.evaluate_ensemble_dict = eval_dict
        self.trial_path=self.get_trial_path()

    def execute_ensemble(self):
        self.setup_trial_dir(self.get_trial_path())
        if 'load_cv_data_from' in self.config.keys():
            self.train_score_ensemble_from_source_cv_data()
        else:
            self.train_score_ensemble_from_new_cv_data()

    def load_json_files(self, model_path, eval_path):
        if not os.path.exists(model_path) or not os.path.exists(eval_path):
            print('Ensemble model or evaluation JSON not found. Exiting...')
            sys.exit(1)
        return (json.load(open(model_path, 'r')),
                json.load(open(eval_path, 'r')))

    def setup_ensemble_agg_evaluation_path(self, ensemble_agg_config):
        eval_path = os.path.join(ensemble_agg_config['models_dir'],
                                 ensemble_agg_config['model_id'])
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)

        self.dump_json(ensemble_agg_config, os.path.join(eval_path, 'evaluate.json'))
        return eval_path

    def get_model_id_for_ensemble_agg(self, agg_method):
        return os.path.join(self.config['ensemble_model_id'], f'evaluation_{agg_method}')

    def call_evaluate_and_plot_for_ensemble(self, scores, ensemble_config):
        plot = EvaluateAndPlot(ensemble_config, scores, self.is_classification)
        plot.plot_all(ensemble_config.get('to_plot', {}))

    def get_and_write_scores_after_ensemble_agg(self, agg_method, eval_path):
        scores = self.combine_scores(agg_method)
        if self.config['save'].get('scores', False):
            scores.to_csv(f'{eval_path}/ensemble_scores_{agg_method}.csv')
        return scores

    def evaluate_ensemble(self):
        for agg_method in self.config['aggregation_method']:
            ensemble_agg_config = self.evaluate_ensemble_dict.copy()
            ensemble_agg_config['model_id'] = self.get_model_id_for_ensemble_agg(agg_method)
            eval_path = self.setup_ensemble_agg_evaluation_path(ensemble_agg_config)
            scores = self.get_and_write_scores_after_ensemble_agg(agg_method, eval_path)
            self.call_evaluate_and_plot_for_ensemble(scores, ensemble_agg_config)

    def train_score_ensemble_from_new_cv_data(self):
        self.create_ensemble_dir_structure()
        self.train_and_score()

    def get_submodel_paths(self, source_path):
        submodels = []
        for d in os.listdir(source_path):
            try:
                int(d)
                submodels.append(d)
            except ValueError:
                pass
        return submodels

    def check_models_dirs_match(self, model_json):
        try:
            assert model_json['models_dir'] == self.config['models_dir']
        except AssertionError:
            print('Ensemble and source models_dir values must match (values below). Exiting...')
            print('Source: {}'.format(model_json['models_dir']))
            print('Ensemble: {}'.format(self.config['models_dir']))
            sys.exit(1)

    def modify_existing_model_json(self, model_dict, d):
        if 'input_changes_by_iteration' in self.config:
            return self.apply_parameters_for_iteration(model_dict, int(d), True)

    def load_and_modify_model_json(self, source_path, d):
        model_json = json.load(open(f'{source_path}/{d}/model.json'))
        self.check_models_dirs_match(model_json)

        model_json['model_cv_to_use'] = model_json['model_id']
        model_json['model_id'] = model_json['model_id'].replace(self.config['load_cv_data_from'],
                                                                self.config['ensemble_model_id'])
        model_json['save']['cv_data'] = True
        model_json = self.modify_existing_model_json(model_json, d)
        return model_json

    def load_and_modify_evaluate_json(self, source_path, d):
        source_eval = f'{source_path}/{d}/evaluate.json'
        if os.path.exists(source_eval):
            eval_json = json.load(open(f'{source_path}/{d}/evaluate.json'))
            assert eval_json['models_dir'] == self.config['models_dir']
            eval_json['model_id'] = eval_json['model_id'].replace(self.config['load_cv_data_from'],
                                                                  self.config['ensemble_model_id'])
            return eval_json
        return None

    def dump_json(self, obj, filepath):
        def cast_np_ints(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        with open(filepath, 'w') as w:
            json.dump(obj, w, indent=3, default=cast_np_ints)

    def copy_and_modify_model_json_from_source_to_destination(self, dest_path, source_path, d):
        model_json = self.load_and_modify_model_json(source_path, d)
        model_path = f'{dest_path}/{d}/model.json'
        self.dump_json(model_json, model_path)

    def copy_and_modify_evaluate_json_from_source_to_destination(self, dest_path, source_path, d):
        eval_json = self.load_and_modify_evaluate_json(source_path, d)
        if eval_json:
            eval_path = f'{dest_path}/{d}/evaluate.json'
            self.dump_json(eval_json, eval_path)

    def copy_and_modify_json_files_from_source_to_destination(self):
        source_path = os.path.join(self.config['models_dir'], self.config['load_cv_data_from'])
        dest_path = os.path.join(self.config['models_dir'], self.config['ensemble_model_id'])
        for d in self.get_submodel_paths(source_path):
            self.copy_and_modify_model_json_from_source_to_destination(dest_path, source_path, d)
            self.copy_and_modify_evaluate_json_from_source_to_destination(dest_path, source_path, d)

    def train_score_ensemble_from_source_cv_data(self):
        for model_nbr in np.arange(self.config['number_of_models']):
            model_path = self.get_model_path(model_nbr)
            self.setup_trial_dir(model_path)
        self.copy_and_modify_json_files_from_source_to_destination()
        self.train_and_score()

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

    def apply_parameters_for_iteration(self, model_dict, model_nbr, preexisting=False):
        # preexisting means that the cv_data was loaded from a preexisting model
        # if that's the case then only features_list, model, and model_params can be changed
        for param_to_change, param_values in self.config['input_changes_by_iteration'].items():
            if (not preexisting) | (param_to_change in ['features_list', 'model', 'model_params']):
                model_dict[param_to_change] = param_values[model_nbr]
        return model_dict

    def apply_seeds(self, model_dict, model_nbr, seed):
        model_dict['dataset_seed'] = int(seed + model_nbr)
        model_dict['fold_seed'] = int(seed + model_nbr)
        return model_dict

    def modify_parameters_for_iteration(self, model_dicts, model_nbr, seed):
        model_dict = model_dicts[model_nbr]
        model_dict = self.apply_seeds(model_dict, model_nbr, seed)
        print(self.config)
        if 'input_changes_by_iteration' in self.config:
            return self.apply_parameters_for_iteration(model_dict, model_nbr)
        else:
            return model_dict

    def dump_evaluation_json_in_submodel(self, model_path, model_id):
        if not os.path.exists(self.config['evaluation_config']):
            print('Warning: evaluate.json file does not exist')
            return

        eval_dict = json.load(open(self.config['evaluation_config']))
        eval_dict['model_id'] = model_id
        eval_dict['ensemble_models'] = self.config['number_of_models']

        json.dump(
            eval_dict,
            open(os.path.join(model_path, 'evaluate.json'), 'w'),
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
        model_eval_path = os.path.join(model_path, 'evaluate.json')
        if os.path.exists(model_eval_path):
            pipeline = ExecuteModelPipeline(model_json_path, model_eval_path, 'Y')
        else:
            pipeline = ExecuteModelPipeline(model_json_path, None, 'Y')
        pipeline.execute_model_pipeline()
        self.set_classification_indicator(pipeline)

    def set_classification_indicator(self, pipeline):
        self.is_classification = pipeline.is_classification

    def train_and_score(self):
        for model_nbr in np.arange(self.config['number_of_models']):
            self.execute_submodel(model_nbr)

    def get_submodel_path(self):
        model_path = os.path.join(self.config['models_dir'],
                                  self.config['ensemble_model_id'])
        for d in os.listdir(model_path):
            try:
                int(d)
                return os.path.join(model_path, d)
            except ValueError:
                pass
        print('No submodel found! Exiting...')
        sys.exit(1)

    def get_labels_tbl(self):
        submodel_path = self.get_submodel_path()
        return json.load(open(os.path.join(submodel_path, 'model.json'), 'r'))['labels_tbl']

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
        score_rename = {c: f'{model_nbr}_{c}'
                        for c in scores.columns
                        if c == 'score' or c.endswith('_score')}
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
        all_scores_nonnull = all_scores_nonnull[~all_scores_nonnull[score_col_base].isnull()]
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
        return all_scores
