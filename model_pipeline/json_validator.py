import json
import os
import pandas as pd
import sys

def validate_model_json(model_dict):
    assert os.path.exists(model_dict['models_dir'])
    assert type(model_dict['index']) is list
    assert type(model_dict['label_col']) in [str, bytes]
    assert type(model_dict['features_tbl']) in [str, bytes]
    assert type(model_dict['features_tbl']) in [str, bytes]
    assert type(model_dict['features_list']) is list
    assert type(model_dict['pos_labels']) is list
    assert type(model_dict['actions']) is dict

    ## assert format is schema.table and that
    ## table exists in hive
    for tbl_str in ['features_tbl', 'features_tbl']:
        schema_and_tbl = model_dict[tbl_str].split('.')
        assert len(schema_and_tbl) == 2
        schema, tbl = schema_and_tbl
        assert os.path.exists(f'data/{schema}/{tbl}.csv')

    ## assert that labels and features share identical index
    features_tbl = pd.read_csv('data/{}/{}.csv'.format(*model_dict['features_tbl'].split('.')))
    labels_tbl = pd.read_csv('data/{}/{}.csv'.format(*model_dict['labels_tbl'].split('.')))
    features_tbl = pd.read_csv('data/{}/{}.csv'.format(*model_dict['features_tbl'].split('.')))

    assert features_tbl.shape[0] == labels_tbl.shape[0]
    assert features_tbl.shape[0] == features_tbl.shape[0]

    feat_cols_set = set(features_tbl.columns)
    label_cols_set = set(labels_tbl.columns)
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
    for label_val in ['pos_labels', 'neg_labels']:
        assert labels_tbl[
                   labels_tbl[model_dict['label_col']].isin(model_dict[label_val])
               ].shape[0] > 0

    ### Cross-validation sets

    ## if this entry is None, make sure the
    ## reference directory exists.
    if model_dict['model_cv_to_use']:
        assert type(model_dict['model_cv_to_use']) in [str, bytes]
        assert os.path.exists(
            os.path.join(model_dict['models_dir'],
                         model_dict['model_cv_to_use'],
                         'cv_data'))
    else:
        ## assert the data structures/types are correct
        assert type(model_dict['fold_seed']) in (int, type(None))
        assert type(model_dict['dataset_seed']) in (int, type(None))
        assert type(model_dict['kfolds']) is int
        assert type(model_dict['strata_cols']) is list
        assert type(model_dict['global_dataset_proportions']) is dict
        assert type(model_dict['dimensional_dataset_proportions']) is dict

        ## assert strata cols are present in the labels table
        assert not set(model_dict['strata_cols']) - label_cols_set

        dataset_types = set(['training', 'holdout', 'throw_away', 'scoring_only'])
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
        for k, dim_list in dim_datasets.items():
            assert (type(dim_list)) is list
            for entry in dim_list:
                assert type(entry) is dict
                assert set(entry.keys()) \
                       == set(['vals', 'dim', 'prop_to_move', 'from_groups'])
                assert entry['dim'] in model_dict['strata_cols']
                assert 0 <= entry['prop_to_move'] <= 1
                assert not set(entry['from_groups']) - dataset_types

    for val in model_dict['dimensional_dataset_proportions'].values():
        for element in val:
            assert element['dim'] in model_dict['strata_cols']

    ### Model Choice
    ## test that model object can be created
    ## from model inputs
    try:
        import importlib

        model_class_str = model_dict['model']
        model_obj_path = '.'.join(model_class_str.split('.')[:-1])
        assert model_obj_path in ['xgboost','sklearn']
        model_name = model_class_str.split('.')[-1]
        model_package = importlib.import_module(model_obj_path)
        model_class = getattr(model_package, model_name)
        _ = model_class(**model_dict['model_params'])
    except Exception as e:
        e

    assert set(model_dict['save'].keys()) \
            == {'cv_data',
                'serialized_models',
                'cv_scores',
                'holdout_scores'}
    set([type(a) for a in model_dict['save'].values()]) == {bool}

    assert set(model_dict['actions'].keys()) \
            == {'do_train_and_score_cv',
                'do_score_holdout',
                'do_evaluate'}
    set([type(a) for a in model_dict['actions'].values()]) == {bool}

    #print(f'Model JSON configuration files passed checks.')

def validate_eval_json(model_dict, plots_dict):
    assert type(plots_dict['label_map']) is dict
    assert type(plots_dict['success_name']) in [str, bytes]
    assert set(plots_dict['label_map'].keys()) == set(['0', '1'])
    assert type(plots_dict['regression_evaluation']) is dict
    assert type(plots_dict['accuracy_at_topn']) is dict

    ### Bins to plot
    ## currently only supports "Bin" and "Percentile"
    assert not set(plots_dict['bin_types']) - set(['Bin', 'Percentile'])
    ## all plot bins values should be ints
    assert plots_dict['plot_bins'] == list(map(int, plots_dict['plot_bins']))
    ## ensure all bins values are in [2, 1000]
    assert list(filter(
        lambda x: 2 <= x <= 1000, plots_dict['plot_bins']
    )) == plots_dict['plot_bins']

    # Threshold plots
    assert type(plots_dict['threshold_metrics']) is list
    # currently only supports Accuracy and F1
    assert not set(plots_dict['threshold_metrics']) - set(['Accuracy','F1'])

    assert set(plots_dict['save'].keys()) \
            == {'data',
                'plots'}
    set([type(a) for a in model_dict['save'].values()]) == {bool}

    if plots_dict['regression_evaluation']:
        assert set(plots_dict['regression_evaluation'].keys()) \
                == {'comparison',
                    'label',
                    'round_score'}
        assert type(plots_dict['regression_evaluation']['round_score']) is bool

    for _, v in plots_dict['accuracy_at_topn'].items():
        assert type(v) is list

    #print(f'Eval dict configuration files passed checks.')