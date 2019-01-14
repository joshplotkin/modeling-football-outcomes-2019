
import argparse
import json
import os
import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

## get spark context with minimal logging
spark = SparkSession.builder.config(
	conf=SparkConf()
	).enableHiveSupport().getOrCreate()
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

assert sum(model_dict['global_dataset_proportions'].values()) == 1
assert type(model_dict['global_dataset_proportions']) is dict
for d in model_dict['dimensional_dataset_proportions'].values():
    assert type(d) is list
    assert sum([type(x) is not dict for x in d]) == 0
    
for k in ['features_list', 'pos_labels', 
          'neg_labels', 'index', 'strata_cols']:
    assert type(model_dict[k]) is list
    
for k in ['global_dataset_proportions',
          'dimensional_dataset_proportions',
          'model_params']:
    assert type(model_dict[k]) is dict

assert set(model_dict['global_dataset_proportions'].keys()) \
        == set(['holdout','throw_away','in_training','scoring_only'])

assert sum(model_dict['global_dataset_proportions'].values()) == 1

label_cols = set(spark.table(model_dict['labels_tbl']).columns)
assert not set(model_dict['index']) - label_cols 
assert not set(model_dict['strata_cols']) - label_cols
assert not set([model_dict['label_col']]) - label_cols
      
feats = set(spark.table(model_dict['features_tbl']).columns)
assert not set(model_dict['features_list']) - feats
assert not set(model_dict['index']) - feats

print 'model.json passed checks.'