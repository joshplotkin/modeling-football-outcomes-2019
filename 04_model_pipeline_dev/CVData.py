from collections import OrderedDict
import numpy as np
import pandas as pd

class CVData:
    def __init__(self, model_dict, spark):
        self.model_dict = model_dict
        self.spark = spark

    def generate_cv_data(self):
        cv_data = self.get_cv_data()
        # global_rolling = self.prop_dict_rolling('dataset')
        # datasets = self.assign_group(cv_data, global_rolling, 'dataset')
        datasets = self.assign_group(cv_data, 'dataset')
        datasets = self.modify_group_for_dim(datasets, 'dataset')

        # assert (1) training set is not empty
        # (2) either k-fold or scoring set is not empty
        assert datasets[datasets['dataset'] == 'in_training'].shape[0] > 0

        training_rows = datasets[datasets['dataset'] == 'in_training']
        scoring_rows = datasets[datasets['dataset'].isin(['scoring_only','holdout'])]

        training_rows = self.assign_k_folds(training_rows)
        training, scoring_only = self.get_training_scoring_sets(training_rows, scoring_rows)

        self.training = training

        self.scoring_only = scoring_only[
            scoring_only['dataset'] == 'scoring_only'
        ].drop('dataset', axis=1)

        self.holdout = scoring_only[
            scoring_only['dataset'] == 'holdout'
        ].drop('dataset', axis=1)

    def write_data(self, training, scoring_only, holdout, write_dir):
        training.to_csv(f'{write_dir}/training.csv', index=False)
        scoring_only.to_csv(f'{write_dir}/scoring_only.csv', index=False)
        holdout.to_csv(f'{write_dir}/holdout.csv', index=False)

    def get_csv_data(self, write_dir=None):
        training = self.training.toPandas()
        scoring_only = self.scoring_only.toPandas()
        holdout = self.holdout.toPandas()
        if write_dir:
            self.write_data(training, scoring_only, holdout, write_dir)
        return {'training': training,
                'scoring_only': scoring_only,
                'holdout': holdout}

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
        label_cols = set(model_dict['index'])\
                     | set(model_dict['strata_cols'])\
                     | set([model_dict['label_col']])
        labels_prep = pd.read_csv('data/{}/{}.csv'
                                  .format(*model_dict['labels_tbl'].split('.')))\
                        .loc[:, label_cols]

        nrows = labels_prep.shape[0]

        np.random.seed(model_dict['dataset_seed'])
        labels_prep['dataset_rnd'] = np.random.random(nrows)
        labels_prep['dim_rnd'] = np.random.random(nrows)

        np.random.seed(model_dict['fold_seed'])
        labels_prep['fold_rnd'] = np.random.random(nrows)

        np.random.seed(None)
        labels_prep['label'] = labels_prep[model_dict['label_col']].map(binarize_label)

        return labels_prep[~labels_prep['label'].isnull()]

    def prop_dict_rolling(self, colname, props=None):
        """given a dictionary of probabilities, where
        the values are floats that sum to 1,
        return a dictionary with the same keys, where
        the values are disjoint windows.
        usage note: top is inclusive. bottom is exclusive unless 0.
        usage note: if both elements are the same, skip
        note: this chooses a random order"""
        if type(props) is type(None):
            props = self.model_dict['global_dataset_proportions']
        rolling_sum = 0
        rolling = OrderedDict()

        shuffled_keys = list(props.keys())
        np.random.seed(self.model_dict[f'{colname}_seed'])
        np.random.shuffle(shuffled_keys)
        np.random.seed(None)

        for k in shuffled_keys:
            v = props[k]
            rolling[k] = (rolling_sum, rolling_sum + v)
            rolling_sum += v
        return rolling

    def assign_group(self, df, colname):
        def assign_value(x, colname):
            if colname == 'dataset':
                rnd_range_mapping = self.prop_dict_rolling(colname)
            else:
                kfolds = self.model_dict['kfolds']
                folds_dict = {k: 1. / kfolds for k in np.arange(kfolds)}
                rnd_range_mapping = self.prop_dict_rolling('fold', folds_dict)
            for i, (k, v) in enumerate(rnd_range_mapping.items()):
                if v[0] < x <= v[1]:
                    return k

        df['pct'] = df.groupby(self.model_dict['strata_cols']) \
                        [f'{colname}_rnd'] \
                        .rank(pct=True)
        df[colname] = df['pct'].apply(lambda x: assign_value(x, colname))
        return df

    def modify_group_for_dim(self, df, colname):
        """given a DF with a groups assigned (variable colname),
        apply a dictionary to post-process the groups according
        to that one dimension. returns original DF with modified
        colname column.
        e.g. move specific seasons to the holdout or throwaway sets.
        """
        dim_props = self.model_dict['dimensional_dataset_proportions'].items()
        for grp, grp_dict_list in dim_props:
            for grp_dict in filter(lambda x: x.get('prop_to_move', 0) > 0, grp_dict_list):
                condition1 = df[colname].isin(grp_dict['from_groups'])
                condition2 = df[grp_dict['dim']].isin(grp_dict['vals'])

                idx = df[(condition1) & (condition2)].index
                df.loc[:, 'pct'] = 99
                df.loc[idx, 'pct'] = df.loc[idx, 'dim_rnd'].rank(pct=True)
                mod_idx = df[df['pct'] <= grp_dict['prop_to_move']].index
                df.loc[mod_idx, colname].value_counts()
                df.loc[mod_idx, colname] = grp
                df.loc[mod_idx, colname].value_counts()
        return df

    def assign_k_folds(self, training_rows):
        """given model dict and pandas DF
        of training data, assign K folds
        using stratified sampling"""
        if self.model_dict['kfolds'] > 1:
            ## make a mapping from fold --> range of random numbers
            ## e.g. fold 0 --> [0.6, 0.8]
            training_rows = self.assign_group(
                training_rows, 'fold'
            )
        else:
            ## if kfold == 1, then skip k-fold.
            ## 1 model will be trained using all data in training dataset.
            ## scoring dataset will be scored with that model.
            training_rows['fold'] = 0

        return training_rows

    def get_training_scoring_sets(self, training_rows, scoring_rows):
        """given model dict and pandas DF of the rows
        to be used for training, and the folds,
        return 2 pandas DFs: prepped training and scoring sets"""
        index = self.model_dict['index']

        features_prep = pd.read_csv(
            'data/{}/{}.csv'.format(*self.model_dict['features_tbl'].split('.'))
        )[set(self.model_dict['features_list']) | set(index)]


        training = features_prep.merge(
            training_rows[index + ['label', 'fold', 'season', 'week_id']],
            on=index
        )
        scoring_only = features_prep.merge(
            scoring_rows[index + ['label','dataset', 'season', 'week_id']],
            on=index
        )

        assert training.shape[0] == training_rows.shape[0]
        assert scoring_only.shape[0] == scoring_rows.shape[0]

        return (training, scoring_only)