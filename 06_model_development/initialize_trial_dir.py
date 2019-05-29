import argparse
import json
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_name', help='trial name (id for incremental #)')
    parser.add_argument('model_config', help='location of model config JSON file')
    parser.add_argument('models_dir', help='models base directory--relative path ok')

    return parser.parse_args()

def get_dir_id(dirs):
    def is_int(d):
        try:
            int(d)
            return True
        except:
            return False

    min_id = min(set(np.arange(100000)) - set(map(int, (filter(is_int, dirs)))))
    return '{:06d}'.format(min_id)

def setup_trial_dir(args, config):
    if args.trial_name == 'id':
        trial_name = get_dir_id(os.listdir(args.models_dir))
    else:
        trial_name = args.trial_name

    trial_dir = os.path.join(args.models_dir, trial_name)
    os.mkdir(trial_dir)
    json.dump(config, open(os.path.join(trial_dir, 'config.json'), 'w'))
    return trial_name

def main():
    args = parse_args()
    config = json.load(open(args.model_config))
    return setup_trial_dir(args, config)

if __name__ == '__main__':
    print(main())
    #TODO: check JSON is valid for my purposes