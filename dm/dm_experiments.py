def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import sys
import time
import copy
sys.path.insert(1, '..')
from utils import get_split_info
from classify import classify
from dm import get_embedding, get_kernel_embedding


def run_experiment(args_og, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios, wl_train_ratios):
    args = copy.deepcopy(args_og)

    result = []
    if args['method'] != 'wl':
        start = time.time()
        x_train, x_val, x_test = get_embedding(args, files_train, run_type='train'), get_embedding(args, files_val,run_type='val'), get_embedding(args, files_test, run_type='test')
        end = time.time()

        for ratio in train_ratios:
            args['train_ratio'] = ratio
            split_point = int(ratio * x_train.shape[0])

            val_score, test_score = classify(args, x_train[:split_point], x_val, x_test, y_train[:split_point], y_val, y_test, run_time=end - start)
            result.append((args, val_score, test_score))
    else:

        for ratio in wl_train_ratios:
            args['train_ratio'] = ratio
            split_point = int(args['train_ratio'] * len(files_train))

            start = time.time()
            x_train, x_val, x_test = get_kernel_embedding(args, files_train[:split_point], files_val, files_test)
            end = time.time()

            val_score, test_score = classify(args, x_train, x_val, x_test, y_train[:split_point], y_val, y_test, run_time=end - start)
            result.append((args, val_score, test_score))

    return result


def run_param_search():
    from config import args as args

    args.update({
        'metric': 'macro-F1',
        'train_ratio': 1.0,
        'val_ratio': 0.1,
        'test_ratio': 0.2,
        'malnet_tiny': False,
    })
    groups = ['type', 'family']

    results = []
    for group in groups:
        args['group'] = group

        files_train, files_val, files_test, y_train, y_val, y_test, label_dict = get_split_info(args)
        args['class_labels'] = list(label_dict.keys())
        args['class_indexes'] = list(label_dict.values())

        for method in ['vnge_slaq', 'lsd_slaq', 'nog', 'feather', 'ldp']:  # , 'sf', 'lsd', 'wl', 'geo_scattering'
            args['method'] = method

            if method == 'wl':
                for n_iter in [2, 5, 10]:
                    args['n_iter'], args['order'], args['n_eigen'], args['n_vectors'], args['n_steps'] = n_iter, 0, 0, 0, 0

                    result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[0.1])
                    results.extend(result)

            elif method == 'feather' or method == 'geo_scattering':
                for order in [4, 5, 6]:
                    args['order'], args['n_iter'], args['n_eigen'], args['n_vectors'], args['n_steps'] = order, 0, 0, 0, 0

                    result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                    results.extend(result)

            elif method == 'sf':
                for n_eigen in [100, 200, 300]:
                    args['n_eigen'], args['order'], args['n_iter'], args['n_vectors'], args['n_steps'] = n_eigen, 0, 0, 0, 0

                    result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                    results.extend(result)

            elif method == 'vnge_slaq' or method == 'lsd_slaq':
                for (n_vectors, n_steps) in [(10, 10), (15, 15), (20, 20)]:
                    args['n_vectors'], args['n_steps'], args['order'], args['n_iter'], args['n_eigen'] = n_vectors, n_steps, 0, 0, 0

                    result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                    results.extend(result)
            else:
                args['n_eigen'], args['order'], args['n_iter'], args['n_vectors'], args['n_steps'] = 0, 0, 0, 0, 0

                result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                results.extend(result)

    for (args_r, val_score, test_score) in results:
        print('Method={}, malnet_tiny={}, group={}, train_ratio={}, n_iter={}, order={}, n_eigen={}, n_vectors={}, n_steps={}, val_{}={}, test_{}={}'.format(
                args_r['method'], args_r['malnet_tiny'], args_r['group'], args_r['train_ratio'], args_r['n_iter'], args['order'],
                args_r['n_eigen'], args_r['n_vectors'], args_r['n_steps'], args_r['metric'], val_score, args_r['metric'], test_score))


def run_best_params():
    from config import args as args

    args.update({
        'metric': 'macro-F1',

        'group': 'type',
        'train_ratio': 1.0,
        'val_ratio': 0.1,
        'test_ratio': 0.2,
        'malnet_tiny': False,
    })

    files_train, files_val, files_test, y_train, y_val, y_test, label_dict = get_split_info(args)
    args['class_labels'] = list(label_dict.keys())
    args['class_indexes'] = list(label_dict.values())

    results = []
    for method in ['vnge_slaq', 'lsd_slaq', 'geo_scattering', 'sf', 'lsd', 'wl', 'nog', 'feather', 'ldp']:
        args['method'] = method

        if method == 'wl':
            for n_iter in [2]:
                args['n_iter'], args['order'], args['n_eigen'], args['n_vectors'], args['n_steps'] = n_iter, 0, 0, 0, 0

                result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                results.extend(result)

        elif method == 'feather' or method == 'geo_scattering':
            for order in [4]:
                args['order'], args['n_iter'], args['n_eigen'], args['n_vectors'], args['n_steps'] = order, 0, 0, 0, 0

                result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                results.extend(result)

        elif method == 'sf':
            for n_eigen in [100]:
                args['n_eigen'], args['order'], args['n_iter'], args['n_vectors'], args['n_steps'] = n_eigen, 0, 0, 0, 0

                result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                results.extend(result)

        elif method == 'vnge_slaq' or method == 'lsd_slaq':
            for (n_vectors, n_steps) in [(10, 10)]:
                args['n_vectors'], args['n_steps'], args['order'], args['n_iter'], args['n_eigen'] = n_vectors, n_steps, 0, 0, 0

                result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
                results.extend(result)
        else:
            args['n_eigen'], args['order'], args['n_iter'], args['n_vectors'], args['n_steps'] = 0, 0, 0, 0, 0

            result = run_experiment(args, files_train, files_val, files_test, y_train, y_val, y_test, train_ratios=[1.0], wl_train_ratios=[1.0])
            results.extend(result)

    for (args_r, val_score, test_score) in results:
        print('Method={}, malnet_tiny={}, group={}, train_ratio={}, n_iter={}, order={}, n_eigen={}, n_vectors={}, n_steps={}, val_{}={}, test_{}={}'.format(
                args_r['method'], args_r['malnet_tiny'], args_r['group'], args_r['train_ratio'], args_r['n_iter'], args['order'],
                args_r['n_eigen'], args_r['n_vectors'], args_r['n_steps'], args_r['metric'], val_score, args_r['metric'], test_score))


if __name__ == '__main__':
    run_param_search()











