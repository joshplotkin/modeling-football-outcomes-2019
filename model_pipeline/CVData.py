from collections import OrderedDict
import itertools
from math import factorial
import numpy as np
import os
import pandas as pd
import sys


class CVData:
    def __init__(self, model_dict):
        self.model_dict = model_dict

    def generate_cv_data(self):
        if self.model_dict['model_cv_to_use']:
            training, scoring_only = self.get_source_data()
        else:
            cv_data = self.get_cv_data()
            datasets = self.assign_group(cv_data, 'dataset')
            datasets = self.modify_group_for_dim(datasets)

            # assert (1) training set is not empty
            # (2) either k-fold or scoring set is not empty
            assert datasets[datasets['dataset'] == 'training'].shape[0] > 0

            training_rows = datasets[datasets['dataset'] == 'training']
            scoring_rows = datasets[datasets['dataset'].isin(['scoring_only','holdout'])]
            training_rows_w_folds = self.assign_group(training_rows, 'fold')
            training, scoring_only = self.get_training_scoring_sets(training_rows_w_folds, scoring_rows)

        if training[self.model_dict['label_col']].unique().shape[0] < 2:
            print('Training data has fewer than 2 classes. Exiting...')
            sys.exit(1)
        if training[self.model_dict['label_col']].unique().shape[0] == 2:
            self.is_classification = True
        else:
            self.is_classification = False


        training = training[[c for c in training.columns if c != 'fold_rnd']]
        scoring_only = scoring_only[[c for c in scoring_only.columns if c != 'fold_rnd']]

        return self.get_and_store_variables(training, scoring_only)

    def write_data(self, training, scoring_only, holdout):
        write_dir = os.path.join(
            self.model_dict['models_dir'],
            self.model_dict['model_id'],
            'cv_data'
        )
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

        self.training.to_csv(f'{write_dir}/training.csv', index=False)
        self.scoring_only.to_csv(f'{write_dir}/scoring_only.csv', index=False)
        self.holdout.to_csv(f'{write_dir}/holdout.csv', index=False)

    def get_and_store_variables(self, training, scoring_only):
            self.training = training

            self.scoring_only = scoring_only[
                scoring_only['dataset'] == 'scoring_only'
            ].drop('dataset', axis=1)

            self.holdout = scoring_only[
                scoring_only['dataset'] == 'holdout'
            ].drop('dataset', axis=1)

            if self.model_dict['save']['cv_data']:
                self.write_data(self.training, self.scoring_only, self.holdout)

            return {'training': self.training,
                    'scoring_only': self.scoring_only,
                    'holdout': self.holdout}

    def get_cv_data(self):
        """using model dict, add random seeds,
        make labels in [0,1], and return only
        relevant columns"""
        def binarize_label(x):
            if x in self.model_dict['pos_labels']:
                return 1
            elif x in self.model_dict['neg_labels']:
                return 0
            return None

        model_dict = self.model_dict
        labels_prep = pd.read_csv('data/{}/{}.csv'
                                  .format(*model_dict['labels_tbl'].split('.')))

        nrows = labels_prep.shape[0]

        np.random.seed(model_dict['dataset_seed'])
        labels_prep['dataset_rnd'] = np.random.random(nrows)
        labels_prep['dim_rnd'] = np.random.random(nrows)

        np.random.seed(model_dict['fold_seed'])
        labels_prep['fold_rnd'] = np.random.random(nrows)

        np.random.seed(None)

        if labels_prep[model_dict['label_col']].unique().shape[0] == 2:
            labels_prep['label'] = labels_prep[model_dict['label_col']].map(binarize_label)
        else:
            labels_prep['label'] = labels_prep[model_dict['label_col']]

        return labels_prep[~labels_prep['label'].isnull()]

    def modify_group_for_dim(self, df):
        """given a DF with a groups assigned ('dataset'),
        apply a dictionary to post-process the groups according
        to that one dimension. returns original DF with modified
        'dataset' column.
        e.g. move specific seasons to the holdout or throwaway sets.
        """
        dim_props = self.model_dict['dimensional_dataset_proportions'].items()
        for grp, grp_dict_list in dim_props:
            for grp_dict in filter(lambda x: x.get('prop_to_move', 0) > 0, grp_dict_list):
                condition1 = df['dataset'].isin(grp_dict['from_groups'])
                condition2 = df[grp_dict['dim']].isin(grp_dict['vals'])

                idx = df[(condition1) & (condition2)].index
                df.loc[:, 'pct'] = 99
                df.loc[idx, 'pct'] = df.loc[idx, 'dim_rnd'].rank(pct=True)
                mod_idx = df[df['pct'] <= grp_dict['prop_to_move']].index
                df.loc[mod_idx, 'dataset'].value_counts()
                df.loc[mod_idx, 'dataset'] = grp
                df.loc[mod_idx, 'dataset'].value_counts()
        return df.drop(['dataset_rnd','dim_rnd','pct'], axis=1)

    def get_assignment_dict_permutations(self, props, purpose):
        if purpose == 'dataset':
            np.random.seed(self.model_dict['dataset_seed'])
        elif purpose == 'fold':
            np.random.seed(self.model_dict['fold_seed'])

        max_permutations = 100000
        if len(props.keys()) > 10:
            import math
            nperm = math.factorial(len(props.keys()))
            sys.stderr.write(f'WARN: due to number of {purpose}s, enumerating {nperm} ' 
                             'permutations is too slow. generating random permutations. '
                             'Blocks might not be as even but it is unlikely this is a problem.')
            all_permutations = [np.random.choice(list(props.keys()), len(props.keys()), replace=False)
                                for _ in np.arange(max_permutations)]
        else:
            all_permutations = list(itertools.permutations(props.keys()))
            np.random.shuffle(all_permutations)

        np.random.seed(None)
        return all_permutations[:max_permutations]

    def get_rolling_dict_given_order(self, keys_ordered, props):
        rolling_sum = 0
        rolling = OrderedDict()
        for k in keys_ordered:
            v = props[k]
            rolling[k] = (rolling_sum, rolling_sum + v)
            rolling_sum += v
        return rolling

    def get_assignment_proportion_dict(self, purpose):
        if purpose == 'dataset':
            props = self.model_dict['global_dataset_proportions']
        elif purpose == 'fold':
            nfolds = self.model_dict['kfolds']
            props = {k: np.round(1/nfolds, 20) for k in np.arange(nfolds)}

        return {k: v for k, v in props.items() if v > 0}

    def get_permutations_of_assignment_proportions(self, purpose):
        props = self.get_assignment_proportion_dict(purpose)
        all_permutations = self.get_assignment_dict_permutations(props, purpose)
        return np.array(
            [self.get_rolling_dict_given_order(perm, props)
             for perm in all_permutations]
        )

    def get_unique_blocks_shuffled(self, cv_data, purpose):
        all_blocks = cv_data[self.model_dict['strata_cols']] \
            .drop_duplicates() \
            .sort_values(by=self.model_dict['strata_cols']) \
            .values
        np.random.seed(self.model_dict[f'{purpose}_seed'])
        np.random.shuffle(all_blocks)
        np.random.seed(None)
        return all_blocks

    def map_block_to_rolling_dict(self, block, block_ids, all_rolling_dicts):
        block_id = block_ids[tuple(block)]
        n_rolling_dicts = len(all_rolling_dicts)
        return all_rolling_dicts[block_id % n_rolling_dicts]

    def map_block_to_dataset_proportion_dict(self, cv_data, all_blocks, all_rolling_dicts):
        block_ids = {tuple(block): i
                     for i, block in enumerate(all_blocks)}
        cv_data['proportion_dict'] = cv_data[
            self.model_dict['strata_cols']
        ].apply(
            lambda block: self.map_block_to_rolling_dict(block, block_ids,
                                                         all_rolling_dicts),
            axis=1
        )
        return cv_data

    def assign_row_to_group(self, x):
        pct, rolling_dict = x
        for i, (k, v) in enumerate(rolling_dict.items()):
            if np.round(v[0], 20) < pct <= np.round(v[1], 20):
                return k
        print('Failed to assign to group: ', pct, rolling_dict)
        sys.exit(1)

    def assign_group(self, cv_data, purpose):
        """purpose is 'dataset' or 'fold', i.e. the groups we are assigning"""
        cv_data['pct'] = cv_data.groupby(self.model_dict['strata_cols']) \
            [f'{purpose}_rnd'] \
            .rank(pct=True)
        all_rolling_dicts = self.get_permutations_of_assignment_proportions(purpose)
        all_blocks = self.get_unique_blocks_shuffled(cv_data, purpose)
        cv_data_w_proportion_dict = self.map_block_to_dataset_proportion_dict(cv_data, all_blocks, all_rolling_dicts)
        cv_data_w_proportion_dict[purpose] = cv_data_w_proportion_dict\
                                                    .loc[:, ['pct', 'proportion_dict']] \
                                                    .apply(self.assign_row_to_group, axis=1)
        return cv_data_w_proportion_dict.drop(['pct','proportion_dict'], axis=1)

    def get_training_scoring_sets(self, training_rows, scoring_rows):
        """given model dict and pandas DF of the rows
        to be used for training, and the folds,
        return 2 pandas DFs: prepped training and scoring sets"""
        index = self.model_dict['index']

        features_prep = pd.read_csv(
            'data/{}/{}.csv'.format(*self.model_dict['features_tbl'].split('.'))
        )[set(self.model_dict['features_list']) | set(index)]

        assert not (set(features_prep.columns.tolist()) \
                    & set(training_rows.columns.tolist())) \
                    - set(index)
        assert not (set(features_prep.columns.tolist()) \
                    & set(scoring_rows.columns.tolist())) \
                    - set(index)

        training = features_prep.merge(
            training_rows,
            on=index
        )
        scoring_only = features_prep.merge(
            scoring_rows,
            on=index
        )

        assert training.shape[0] == training_rows.shape[0]
        assert scoring_only.shape[0] == scoring_rows.shape[0]

        return (training, scoring_only)

    def get_source_data(self):
        """instead of re-computing a CV set,
        use one from a reference model. tests
        that the index in the reference data
        is a subset of the current data's index."""
        ref_model_path = os.path.join(
            self.model_dict['models_dir'],
            self.model_dict['model_cv_to_use'],
            'cv_data'
        )
        if not os.path.exists(ref_model_path):
            print('Reference data (path below) not available. Exiting...\n' + source_dir)
            sys.exit(1)

        cv_data_files = list(filter(lambda x: '.csv' in x, os.listdir(ref_model_path)))

        index = self.model_dict['index']
        all_data = pd.read_csv(
                'data/{}/{}.csv'.format(*self.model_dict['features_tbl'].split('.'))
            )[set(self.model_dict['features_list']) | set(index)]

        for i, f in enumerate(cv_data_files):
            subset_data = pd.read_csv(os.path.join(ref_model_path, f))
            fields = list(set(subset_data.columns.tolist()) - set(all_data.columns.tolist()))
            fields.extend(index)

            subset_data = subset_data[fields]
            # check that these CV sets' indexes are all represented
            # in the cv_data Spark DF
            assert subset_data.merge(
                    all_data, left_on=index, right_on=index
                ).shape[0] == subset_data.shape[0]

            curr_source_data = subset_data.merge(all_data, left_on=index, right_on=index)
            curr_source_data['dataset'] = f.split('.')[0]
            if i == 0:
                source_data = curr_source_data
            else:
                source_data = source_data.append(curr_source_data).reset_index(drop=True)

        training = source_data[source_data['dataset'] == 'training']
        scoring_only = source_data[source_data['dataset'].isin(['scoring_only', 'holdout'])]
        return training, scoring_only