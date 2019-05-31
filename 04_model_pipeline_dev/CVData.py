import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window

class CVData:
    def __init__(self, model_dict, spark):
        self.model_dict = model_dict
        self.spark = spark

    def generate_cv_data(self):
        cv_data = self.get_cv_data()
        global_rolling = self.prop_dict_rolling(self.model_dict['global_dataset_proportions'])
        datasets = self.assign_group(cv_data, global_rolling, 'dataset')
        datasets = self.modify_group_for_dim(datasets, 'dataset')

        # assert (1) training set is not empty
        # (2) either k-fold or scoring set is not empty
        assert datasets.filter(col('dataset') == 'in_training').count() > 0
        if self.model_dict['kfolds'] <= 1:
            assert datasets.filter(col('dataset') == 'scoring_only').count() > 0

        training_rows = datasets.filter(col('dataset') == 'in_training')
        scoring_rows = datasets.filter(col('dataset').isin(['scoring_only','holdout']))

        training_rows = self.assign_k_folds(training_rows)
        training, scoring_only = self.get_training_scoring_sets(training_rows, scoring_rows)

        self.training = training
        self.scoring_only = scoring_only\
                                .filter(col('dataset') == 'scoring_only')\
                                .drop('dataset')
        self.holdout = scoring_only\
                                .filter(col('dataset') == 'holdout')\
                                .drop('dataset')

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
        model_dict = self.model_dict
        labels_prep = self.spark.table(model_dict['labels_tbl']).select(
            *(set(model_dict['index'])
              | set(model_dict['strata_cols'])
              | set([model_dict['label_col']]))
        )

        return labels_prep.withColumn(
            'dataset_rnd', F.rand(model_dict['dataset_seed'])
        ).withColumn(
            'dim_rnd', F.rand(model_dict['dataset_seed'])
        ).withColumn(
            'kfold_rnd', F.rand(model_dict['kfold_seed'])
        ).withColumn(
            'label',
            F.when(
                col(model_dict['label_col']).isin(model_dict['pos_labels']),
                1
            ).when(
                col(model_dict['label_col']).isin(model_dict['neg_labels']),
                0
            ).otherwise(None)
        ).filter(
            col('label').isNotNull()
        )

    def prop_dict_rolling(self, d):
        """given a dictionary of probabilities, where
        the values are floats that sum to 1,
        return a dictionary with the same keys, where
        the values are disjoint windows.
        usage note: top is inclusive. bottom is exclusive unless 0.
        usage note: if both elements are the same, skip"""
        rolling_sum = 0
        rolling = {}
        for k, v in d.items():
            rolling[k] = (rolling_sum, rolling_sum + v)
            rolling_sum += v
        return rolling

    def assign_group(self, df, d, colname):
        """given (1) a dictionary of ranges,
        (2) a DF with random values ranked
        by random block, and
        (3) a name for the grouped columns,
        return DF with a new column that
        assigns group membership"""
        strata_cols = self.model_dict['strata_cols']
        window = Window.orderBy('dataset_rnd') \
            .partitionBy(*self.model_dict['strata_cols'])
        df = df.withColumn('dataset_rk',
                           F.percent_rank().over(window))
        for i, (k, v) in enumerate(d.items()):
            ## if the bottom is 0, make it -1 to include 0
            min_val = -1 if v[0] == 0 else min_val
            if type(k) is np.int64:
                k = int(k)
            if i == 0:
                group_assign_cond = F.when(
                    (col('dataset_rk') > min_val)
                    & (col('dataset_rk') <= v[1]),
                    F.lit(k)
                )
            else:
                group_assign_cond = group_assign_cond.when(
                    (col('dataset_rk') > min_val)
                    & (col('dataset_rk') <= v[1]),
                    F.lit(k)
                )

        return df.withColumn(colname, group_assign_cond)

    def modify_group_for_dim(self, df, colname):
        """given a DF with a groups assigned (variable colname),
        apply a dictionary to post-process the groups according
        to that one dimension. returns original DF with modified
        colname column.
        e.g. move specific seasons to the holdout or throwaway sets.
        """
        dim_props = self.model_dict['dimensional_dataset_proportions'].items()
        for grp, grp_dict_list in dim_props:
            for grp_dict in grp_dict_list:
                window = Window.orderBy('dim_rnd') \
                    .partitionBy(grp_dict['dim'], colname)
                df = df.withColumn('dim_rk', F.percent_rank().over(window))

                ## if (1) the column is within the set values,
                ## (2) the pre-existing group falls within those set values, and
                ## (3) the random value is below the set threshold,
                ## then override and modify the group membership
                if grp_dict['prop_to_move'] > 0:
                    df = df.withColumn(
                        colname,
                        F.when(
                            (col(grp_dict['dim']).isin(grp_dict['vals']))
                            & (col(colname).isin(grp_dict['from_groups']))
                            & (col('dim_rk') >= 1 - grp_dict['prop_to_move']),
                            grp
                        ).otherwise(col(colname))
                    )
        return df

    def assign_k_folds(self, training_rows):
        '''given model dict and pandas DF
        of training data, assign K folds
        using stratified sampling
        '''
        ## assign K folds
        kfolds = self.model_dict['kfolds']
        if kfolds > 1:
            ## make a mapping from fold --> range of random numbers
            ## e.g. fold 0 --> [0.6, 0.8]
            folds_dict = {k: 1. / kfolds for k in np.arange(kfolds)}
            folds_rolling = self.prop_dict_rolling(folds_dict)
            ## apply mapping
            training_rows = self.assign_group(
                training_rows, folds_rolling, 'fold'
            )
        else:
            ## if kfold == 1, then skip k-fold.
            ## 1 model will be trained using all data in training dataset.
            ## scoring dataset will be scored with that model.
            training_rows = training_rows.withColumn('fold', F.lit(None))

        return training_rows

    def get_training_scoring_sets(self, training_rows, scoring_rows):
        '''given model dict and pandas DF of the rows
        to be used for training, and the folds,
        return 2 pandas DFs: prepped training and scoring sets'''
        index = self.model_dict['index']
        features_prep = self.spark.table(self.model_dict['features_tbl']).select(
            *(set(self.model_dict['features_list'])
              | set(index))
        )

        training = features_prep.join(
            training_rows.select(*(index + ['label', 'fold', 'season', 'week_id'])),
            on=index
        )
        scoring_only = features_prep.join(
            scoring_rows.select(*(index + ['label','dataset', 'season', 'week_id'])),
            on=index
        )

        assert training.count() == training_rows.count()
        assert scoring_only.count() == scoring_rows.count()

        return (training, scoring_only)