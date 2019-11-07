import os
import yaml

from francis.tensorflow_tools import read_summary


def load_configs(path):
    runs = os.listdir(path)
    configs = dict()
    for r in runs:
        config_path = os.path.join(path, r, 'data/config.yaml')
        try:
            with open(config_path) as f:
                configs[r] = yaml.load(f)
        except:
            pass
    return configs


def match_configs_against(configs, query_function, min_iteration=-1, min_date=None, root_path=''):
    runs = []
    for r, c in configs.items():
        if query_function(c):
            if min_date is not None and os.path.getmtime(os.path.join(root_path, r)) < min_date.timestamp():
                continue
            if min_iteration > 0:
                summary = read_summary(os.path.join(root_path, r, 'results'))
                keys = list(summary.keys())
                if len(keys) == 0:
                    continue
                last_iter = summary[keys[-1]][-1][0]
                if last_iter < min_iteration:
                    continue
            runs.append(r)
            
    return runs

