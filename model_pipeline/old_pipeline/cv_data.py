import argparse
import cPickle as pickle
import json
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *

## get spark context with minimal logging
spark = SparkSession\
            .builder\
            .config(conf=SparkConf())\
            .enableHiveSupport()\
            .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

## mode ID is only command-line argument
## store as variable model_id and assert
## that path models/model_id exists,
## then change to that directory
parser = argparse.ArgumentParser()
parser.add_argument('model_id', help='model ID/directory name')
args = parser.parse_args()
model_id = args.model_id 
assert os.path.exists('models/{}'.format(model_id))
os.chdir('models/{}'.format(model_id))

## load model configuration file
model_dict = json.load(open('model.json','r'))

## default schema
spark.sql('USE football_games')

### GLOBAL FUNCTIONS
def get_cv_data(model_dict):
    '''using model dict, add random seeds, 
    make labels in [0,1], and return only 
    relevant columns'''
    labels_prep = spark.table(model_dict['labels_tbl']).select(
            *( set(model_dict['index']) 
              | set(model_dict['strata_cols']) 
              | set([model_dict['label_col']]) )
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

def prop_dict_rolling(d):
    '''given a dictionary of probabilities, where
    the values are floats that sum to 1, 
    return a dictionary with the same keys, where
    the values are disjoint windows.
    usage note: top is inclusive. bottom is exclusive unless 0.
    usage note: if both elements are the same, skip'''
    rolling_sum = 0
    rolling = {}
    for k,v in d.iteritems():
        rolling[k] = (rolling_sum, rolling_sum+v)
        rolling_sum += v
    return rolling

def modify_group_for_dim(model_dict, df, d, colname):
    '''given a DF with a groups assigned (variable colname),
    apply a dictionary to post-process the groups according 
    to that one dimension. returns original DF with modified
    colname column.
    e.g. move specific seasons to the holdout or throwaway sets.
    '''
    dim_props = model_dict['dimensional_dataset_proportions'].iteritems()
    for grp, grp_dict_list in dim_props:
        for grp_dict in grp_dict_list:
            window = Window.orderBy('dim_rnd')\
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

def assign_k_folds(model_dict, training_rows):
    '''given model dict and pandas DF
    of training data, assign K folds
    using stratified sampling
    '''
    ## assign K folds
    kfolds = model_dict['kfolds']
    if kfolds > 1:
        ## make a mapping from fold --> range of random numbers
        ## e.g. fold 0 --> [0.6, 0.8]
        folds_dict = {k: 1./kfolds for k in np.arange(kfolds)}
        folds_rolling = prop_dict_rolling(folds_dict)
        ## apply mapping
        training_rows = assign_group(
                model_dict, training_rows, folds_rolling, 
                model_dict['strata_cols'], 'fold'
            )
    else:
        ## if kfold == 1, then skip k-fold.
        ## 1 model will be trained using all data in training dataset.
        ## scoring dataset will be scored with that model.
        training_rows = training_rows.withColumn('fold', F.lit(None))
        
    return training_rows

def assign_group(model_dict, df, d, strata_cols, colname):
    '''given (1) a dictionary of ranges,
    (2) a DF with random values ranked 
    by random block, and 
    (3) a name for the grouped columns,
    return DF with a new column that 
    assigns group membership'''
    window = Window.orderBy('dataset_rnd')\
                   .partitionBy(*model_dict['strata_cols'])
    df = df.withColumn('dataset_rk', 
                       F.percent_rank().over(window))
    for i, (k,v) in enumerate(d.iteritems()):
        ## if the bottom is 0, make it -1 to include 0
        min_val = -1 if v[0] == 0 else min_val
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

def prep_datasets(model_dict, data_rows):
    '''given model dict and pandas DF of the rows
    (or dict of rows) to be used for training, and the folds, 
    return 2 pandas DFs: prepped training and scoring sets'''
    index = model_dict['index']
    features_prep = spark.table(model_dict['features_tbl']).select(
            *( set(model_dict['features_list']) 
               | set(index) )
        )
    features_list = model_dict['features_list']
    
    if type(data_rows) is dict:
        for k, rows_sdf in data_rows.iteritems():
            if 'fold' in rows_sdf.columns:
                addon_cols = ['label','fold']
            else:
                addon_cols = ['label']
            prepped_sdf = features_prep.join(
                rows_sdf.select(*(index + addon_cols)),
                on=index
            ).select(
                *(index + addon_cols + features_list)
            )
            assert prepped_sdf.count() == rows_sdf.count()
            data_rows[k] = prepped_sdf

        return data_rows
    else:
        if 'fold' in data_rows.columns:
            addon_cols = ['label','fold']
        else:
            addon_cols = ['label']
        prepped_sdf = features_prep.join(
            data_rows.select(*(index + addon_cols)),
            on=index
        ).select(
            *(index + addon_cols + features_list)
        )
        assert prepped_sdf.count() == data_rows.count()
        return prepped_sdf

def get_test_reference_data(model_dict, cv_data):
    '''instead of re-computing a CV set,
    use one from a reference model. tests
    that the index in the reference data
    is a subset of the current data's index.'''
    ref_datasets = {}
    feats = sorted(model_dict['features_list'])
    
    idx = model_dict['index']
    ref_model_path = '../{}/cv_data'.format(model_dict['model_cv_to_use'])
    files = os.listdir(ref_model_path)
    
    all_data = prep_datasets(model_dict, cv_data).toPandas()
    
    for f in filter(lambda x: '.csv' in x, files):
        subset_data = pd.read_csv('{}/{}'.format(ref_model_path, f))
        if 'fold' in subset_data.columns:
            fields = idx + ['fold']
        else:
            fields = idx
        
        subset_data = subset_data[fields]
        ## check that these CV sets' indexes are all represented
        ## in the cv_data Spark DF
        assert subset_data.merge(
                all_data, left_on=idx, right_on=idx
            ).shape[0] == subset_data.shape[0]

        ref_datasets[f] = subset_data.merge(
                all_data, left_on=idx, right_on=idx
            )
    return ref_datasets

## START EXECUTION
## writing to Hive tables is more scalable but writing to csv
## is faster on smaller data (no overhead of spinning up sparkcontext)
if not os.path.exists('cv_data'): 
    os.mkdir('cv_data')

cv_data = get_cv_data(model_dict)

## don't re-compute the CV datasets. load from a pre-existing model
if model_dict['model_cv_to_use']:
    ## get the data and make sure the indexes overlap
    ref_datasets = get_test_reference_data(model_dict, cv_data)
        
    required_data = ['training.csv','scoring_only.csv']
    if model_dict['holdout_set']['store_to_disk'] is True:
        required_data.append('holdout.csv')
    ## assert that all needed CV data is available
    assert set(required_data) == set(ref_datasets.keys())

    for k,v in ref_datasets.iteritems():
        v[model_dict['index'] 
          + model_dict['features_list']
         ].to_csv('cv_data/{}'.format(k), index=False)

## compute CV datasets
else:
    ## assert that the "label" column created in get_cv_data is binary
    assert cv_data.groupby('label').count().toPandas().shape[0] == 2
    global_rolling = prop_dict_rolling(
        model_dict['global_dataset_proportions']
    )
    datasets = assign_group(
        model_dict, cv_data, global_rolling, 
        model_dict['strata_cols'], 'dataset'
    )
    datasets = modify_group_for_dim(
        model_dict, datasets, 
        model_dict['dimensional_dataset_proportions'], 'dataset' 
    )

    ## assert (1) training set is not empty 
    ## (2) either k-fold or scoring set is not empty
    assert datasets.filter(col('dataset') == 'in_training').count() > 0
    if model_dict['kfolds'] <= 1:
        assert datasets.filter(col('dataset') == 'scoring_only').count() > 0

    data_rows = {
        'scoring_only': datasets.filter(col('dataset') == 'scoring_only'),
        'training': datasets.filter(col('dataset') == 'in_training')
    }
    ## model.json indicates whether to store holdout set
    if model_dict['holdout_set']['store_to_disk'] is True:
        data_rows['holdout'] = datasets.filter(col('dataset') == 'holdout')

    data_rows['training'] = assign_k_folds(model_dict, data_rows['training'])
    data_rows = prep_datasets(model_dict, data_rows)

    for dataset_name, dataset_sdf in data_rows.iteritems():
        dataset_sdf\
            .toPandas()\
            .to_csv('cv_data/{}.csv'.format(dataset_name), index=False)

print 'cv sets wrote successfully.'