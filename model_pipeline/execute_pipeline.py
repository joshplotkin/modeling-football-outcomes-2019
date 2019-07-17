# TODO: CleanseDataTODO.py
# TODO later: perform independent tasks like evaluation on an existing directory
# TODO: fix plots... things that don't look good; also with large number of folds, no need for so many ROC curves
# TODO: if number of folds > some number (50?), do LOO and make each row a fold
#

import argparse

from ExecuteModelPipeline import ExecuteModelPipeline

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_config', help='location of model config JSON file')
    parser.add_argument('eval_config', help='location of evaluation config JSON file (allowed to be nonexistent/null)')
    parser.add_argument('overwrite', help='Y/N: overwrite if trial name already has a directory (defaults to N)')
    return parser.parse_args()


def main():
    args = setup_argparse()
    ExecuteModelPipeline(args.model_config, args.eval_config, args.overwrite)


if __name__ == '__main__':
    """
    Example (run from modeling-football-games directory):
    python model_pipeline/execute_pipeline.py 04_model_pipeline_dev/model.json 04_model_pipeline_dev/evaluate.json N
    """
    main()
