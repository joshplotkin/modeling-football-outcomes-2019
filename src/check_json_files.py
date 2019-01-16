
import argparse
import json
import os
import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col

## get spark context with minimal logging
spark = SparkSession.builder\
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

## load model and plots configuration files
model_dict = json.load(open('model.json','r'))
plots_dict = json.load(open('plots.json','r'))

#### START CHECKS

### Hive Tables

## assert these fields are of the correct type
assert type(model_dict['index']) is list
assert type(model_dict['label_col']) in [str, unicode]
assert type(model_dict['features_tbl']) in [str, unicode]
assert type(model_dict['features_tbl']) in [str, unicode]
assert type(model_dict['features_list']) is list
assert type(model_dict['pos_labels']) is list

## assert format is schema.table and that
## table exists in hive
for tbl_str in ['features_tbl','features_tbl']:
    schema_and_tbl = model_dict[tbl_str].split('.')
    assert len(schema_and_tbl) == 2
    schema, tbl = schema_and_tbl
    assert spark.sql(
            'show tables in {}'.format(schema)
        ).filter(
            col('tableName') == tbl
        ).count() == 1

feat_cols_set = set(spark.table(model_dict['features_tbl']).columns)
label_cols_set = set(spark.table(model_dict['labels_tbl']).columns)
idx_set = set(model_dict['index'])
feat_set = set(model_dict['features_list'])
label_set = set([model_dict['label_col']])

## assert the chosen columns exist in the
## chosen tables
assert not idx_set - feat_cols_set
assert not idx_set - label_cols_set
assert not feat_set - feat_cols_set
assert not label_set - label_cols_set

## check that positive and negative label values 
## are valid
for label_val in ['pos_labels','neg_labels']:
    assert spark.table(
            model_dict['labels_tbl']
        ).filter(
            col(model_dict['label_col']).isin(model_dict[label_val])
        ).count() > 0

### Cross-validation sets

## if this entry is None, make sure the 
## reference directory exists.
if model_dict['model_cv_to_use']:
    assert type(model_dict['model_cv_to_use']) in [str, unicode]
    assert os.path.exists('../{}'.format(model_dict['model_cv_to_use']))
else:
    ## assert the data structures/types are correct
    assert type(model_dict['kfold_seed']) is int
    assert type(model_dict['dataset_seed']) is int
    assert type(model_dict['kfolds']) is int
    assert type(model_dict['strata_cols']) is list
    assert type(model_dict['global_dataset_proportions']) is dict
    assert type(model_dict['dimensional_dataset_proportions']) is dict
    assert type(model_dict['holdout_set']) is dict

    ## assert strata cols are present in the labels table
    assert not set(model_dict['strata_cols']) - label_cols_set

    dataset_types = set(['in_training','holdout','throw_away','scoring_only'])
    global_datasets = model_dict['global_dataset_proportions']
    dim_datasets = model_dict['dimensional_dataset_proportions']

    ## assert global_dataset_proportions has all possible dataset types
    assert set(global_datasets.keys()) == dataset_types
    ## values are proportions that must sum to 1
    assert sum(global_datasets.values()) == 1
    ## assert that the keys are valid dataset types
    assert not set(dim_datasets.keys()) - dataset_types
    ## assert the following (in order of assertion block):
    ## (1) each value is a list
    ## (2) each element of the list is a dict
    ## (3) each dict has the 5 required keys
    ## (4) the "dim" field is in the strata columns 
    ## (5) "prop_to_move" field is [0, 1]
    ## (6) "from_groups" are in the possible dataset types
    for k, dim_list in dim_datasets.iteritems():
        assert (type(dim_list)) is list
        for entry in dim_list:
            assert type(entry) is dict
            assert set(entry.keys()) \
                    == set(['vals','dim','prop_to_move','from_groups'])
            assert entry['dim'] in model_dict['strata_cols']
            assert 0 <= entry['prop_to_move'] <= 1
            assert not set(entry['from_groups']) - dataset_types

    ## assert holdout set has 2 keys (store_to_disk, score_using_full_model)
    ## and the corresponding values are boolean
    assert set(model_dict['holdout_set'].keys()) \
            == set(['store_to_disk','score_using_full_model'])
    assert len(filter(
        lambda x: type(x) is not bool, 
        model_dict['holdout_set'].values()
    )) == 0
    ## if holdout data isn't stored, it can't be scored
    assert not (model_dict['holdout_set']['store_to_disk'] is False) \
                & (model_dict['holdout_set']['score_using_full_model'] is True)

### Model Choice
## test that model object can be created
## from model inputs
try:
    import importlib

    model_class_str = model_dict['model']
    model_obj_path = '.'.join(model_class_str.split('.')[:-1])
    model_name = model_class_str.split('.')[-1]
    model_package = importlib.import_module(model_obj_path)
    model_class = getattr(model_package, model_name)
    _ = model_class(**model_dict['model_params'])
except Exception as e:
    e

### Plot Labels
assert type(plots_dict['label_map']) is dict
assert type(plots_dict['success_name']) in [str, unicode]
assert set(plots_dict['label_map'].keys()) == set(['0','1'])

### Bins to plot
## currently only supports "Bin" and "Percentile"
assert not set(plots_dict['bin_types']) - set(['Bin','Percentile'])
## all plot bins values should be ints
assert plots_dict['plot_bins'] == map(int, plots_dict['plot_bins'])
## ensure all bins values are in [2, 1000]
assert filter(
        lambda x: 2 <= x <= 1000, plots_dict['plot_bins']
    )   == plots_dict['plot_bins']

### Threshold plots
assert type(plots_dict['threshold_metrics']) is list
## currently only supports Accuracy and F1
assert not set(plots_dict['threshold_metrics']) - set(['Accuracy','F1'])

print 'JSON configuration files passed checks.'