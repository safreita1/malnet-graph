import os
import multiprocessing

dm_args = {
    'method': 'vnge',

    'group': 'family',
    'train_ratio': 1.0,
    'val_ratio': 0.1,
    'test_ratio': 0.2,

    'seed': 0,
    'shuffle': True,
    'n_cores': multiprocessing.cpu_count(),

    'log_comment': 'no comment',
    'graph_dir': '',  # *** Add your own path here ***
    'results_dir': os.path.dirname(os.path.abspath("__file__")) + '/results/'

}

os.makedirs(dm_args['results_dir'], exist_ok=True)


