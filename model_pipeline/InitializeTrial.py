import json
from numpy import arange
import os
import sys

class InitializeTrial:
    def __init__(self, model_config, eval_config, overwrite='N'):
        model_json, eval_json = self.load_json_files(model_config, eval_config)
        trial_name = self.get_trial_name(model_json['models_dir'], model_json['model_id'])
        model_json['model_id'] = trial_name

        trial_path = os.path.join(model_json['models_dir'], trial_name)
        self.setup_trial_dir(overwrite, trial_path)
        self.dump_json_to_trial_path(trial_path, model_json, eval_json)

        self.model_dict = model_json
        self.evaluation_dict = eval_json


    def load_json_files(self, model_config, eval_config):
        def check_path_exists(path):
            if not os.path.exists(path):
                print(f'{path} does not exist. exiting...')
                sys.exit(1)

        check_path_exists(model_config)
        model_json = json.load(open(model_config, 'r'))
        check_path_exists(model_json['models_dir'])

        if eval_config:
            if os.path.exists(eval_config):
                eval_json = json.load(open(eval_config, 'r'))
            else:
                eval_json = None
        else:
            eval_json = None

        return model_json, eval_json

    def get_dir_id(self, dirs):
        def is_int(d):
            try:
                int(d)
                return True
            except:
                return False

        min_id = min(set(arange(100000)) - set(map(int, (filter(is_int, dirs)))))
        return '{:06d}'.format(min_id)

    def get_trial_name(self, models_dir, trial_name_from_model_json):
        if trial_name_from_model_json == 'id':
            trial_name = self.get_dir_id(os.listdir(models_dir))
        else:
            trial_name = trial_name_from_model_json
        return trial_name

    def setup_trial_dir(self, overwrite, trial_path):
        overwrite = True if overwrite.upper()[0] == 'Y' else False

        print('Model Path:\n{}'.format(trial_path))
        if (overwrite is False) & (os.path.exists(trial_path)):
            print('model path already exists and user input disallows overwriting. exiting...')
            sys.exit(1)

        if (overwrite) & (os.path.exists(trial_path)):
            import shutil
            shutil.rmtree(trial_path)
        os.mkdir(trial_path)

    def dump_json_to_trial_path(self, trial_path, model_json, eval_json):
        json.dump(model_json, open(os.path.join(trial_path, 'model.json'), 'w'))

        if type(eval_json) is not type(None):
            json.dump(eval_json, open(os.path.join(trial_path, 'evaluate.json'), 'w'))


