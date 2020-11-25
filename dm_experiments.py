def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import gc
import time
import itertools
import numpy as np
from tqdm import tqdm
from pprint import pprint
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from seb import seb
from grakel import GraphKernel
from slaq import netlsd, vnge, vnge_naive
from karate import feather, fgsd, sf, geo_scattering
from utils import process_file_for_karate, process_file_for_slaq, process_file_for_seb, process_file_for_grakel, get_data\
    , split_rand, save_split_info, kernel_transform, process_file_for_nog


def save_report(args, params, report, macro_f1, run_time):
    save_path = args['results_dir'] + '{}/{}/{}/{}/info.txt'.format(args['group'], args['train_ratio'], args['method'], args['log_comment'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        pprint(args, stream=f)
        pprint('\n\n', stream=f)
        pprint(params, stream=f)
        pprint(report, stream=f)
        pprint('Macro-f1: {}'.format(macro_f1), stream=f)
        pprint('Malware group: {}'.format(args['group']), stream=f)
        pprint('Train ratio: {}'.format(args['train_ratio']), stream=f)
        pprint('Embedding took {} seconds w/ {} cpu cores'.format(round(run_time, 2), args['n_cores']), stream=f)
    

def run_method(file, method):
    if method == 'lsd':
        graph = process_file_for_slaq(file)
        result = netlsd(graph)

    elif method == 'vnge':
        graph = process_file_for_slaq(file)
        result = vnge(graph)

    elif method == 'vnge_naive':
        graph = process_file_for_slaq(file)
        result = vnge_naive(graph)

    elif method == 'feather':
        graph = process_file_for_karate(file)
        result = feather(graph)

    elif method == 'fgsd':
        graph = process_file_for_karate(file)
        result = fgsd(graph)

    elif method == 'sf':
        graph = process_file_for_karate(file)
        result = sf(graph)

    elif method == 'geo_scattering':
        graph = process_file_for_karate(file)
        result = geo_scattering(graph)

    elif method == 'seb':
        subgraphs = process_file_for_seb(file)
        result = seb(subgraphs)

    elif method == 'nog':
        graph = process_file_for_nog(file)
        result = np.array([graph.number_of_nodes(), graph.number_of_edges()], dtype=np.int64)

    else:
        print('Method {} not implemented'.format(method))
        exit(1)

    return result


def get_kernel_embedding(args, train_files, val_files, test_files):
    print('\n******Running WL Kernel on train set******')
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5}, "subtree_wl"], normalize=True, n_jobs=args['n_cores'])

    graphs = Parallel(n_jobs=args['n_cores'])(
        delayed(process_file_for_grakel)(file)
        for file in tqdm(train_files))
    x_train = gk.fit_transform(graphs)

    print('\n******Running WL Kernel on val set******')
    x_val = kernel_transform(args, val_files, gk)

    print('\n******Running WL Kernel on test set******')
    x_test = kernel_transform(args, test_files, gk)

    return x_train, x_val, x_test


def get_embedding(args, files, run_type):
    print('\n******Running {} on {} set******'.format(args['method'], run_type))

    embedding = Parallel(n_jobs=args['n_cores'])(
        delayed(run_method)(file, args['method'])
        for file in tqdm(files))

    embedding = np.asarray(embedding)
    if len(embedding.shape) == 1: embedding = embedding.reshape(-1, 1)

    return embedding


def grid_search(args, x_train, y_train, x_val, y_val):
    best_macro_f1 = 0

    n_estimators = [1, 5, 10, 50]
    max_depths = [1, 5, 10, 20]
    params = list(itertools.product(n_estimators, max_depths))

    for n_estimator, max_depth in tqdm(params):
        clf = RandomForestClassifier(random_state=args['seed'], n_estimators=n_estimator, max_depth=max_depth, n_jobs=args['n_cores'])
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_val)
        score = f1_score(y_val, y_pred, average='macro')

        if score > best_macro_f1:
            best_macro_f1 = score
            best_params = {'n_estimators': n_estimator, 'max_depth': max_depth}

        del clf
        gc.collect()

    print('Best val macro F1 score:', best_macro_f1)
    print('Best params:', best_params)

    return best_params


def classify(args, x_train, x_val, x_test, y_train, y_val, y_test, label_dict):
    print('Classifying {} graph embeddings...'.format(args['method']))
    print('Number of train samples: {}, val samples: {}, test samples: {}'.format(x_train.shape[0], x_val.shape[0], x_test.shape[0]))

    # standardize the data
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_val, x_test = scalar.transform(x_val), scalar.transform(x_test)

    params = grid_search(args, x_train, y_train, x_val, y_val)

    clf = RandomForestClassifier(random_state=args['seed'], n_jobs=args['n_cores'])
    clf.set_params(**params)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    score = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=list(label_dict.keys()), labels=list(label_dict.values()))

    return report, score, params


def run_all_methods(args, limit):
    train_ratios = [1.0]

    files, labels, label_dict = get_data(args, limit=limit)
    train_idx, val_idx, test_idx = split_rand(labels, train_ratio=args['train_ratio'], val_ratio=args['val_ratio'], test_ratio=args['test_ratio'],
                                              seed=args['seed'], shuffle=args['shuffle'])
    save_split_info(args, files, train_idx, val_idx, test_idx)

    files_train, y_train = np.asarray(files)[train_idx].tolist(), np.asarray(labels)[train_idx]
    files_val, y_val = np.asarray(files)[val_idx].tolist(), np.asarray(labels)[val_idx]
    files_test, y_test = np.asarray(files)[test_idx].tolist(), np.asarray(labels)[test_idx]

    args['method'] = 'nog'
    start = time.time()
    x_train, x_val, x_test = get_embedding(args, files_train, run_type='train'), get_embedding(args, files_val, run_type='val'), get_embedding(args, files_test, run_type='test')
    end = time.time()
    for ratio in train_ratios:
        args['train_ratio'] = ratio
        split_point = int(ratio*x_train.shape[0])

        report, mf1, params = classify(args, x_train[0:split_point], x_val, x_test, y_train[0:split_point], y_val, y_test, label_dict)
        save_report(args, params, report, mf1, end-start)

    args['method'] = 'vnge'
    start = time.time()
    x_train, x_val, x_test = get_embedding(args, files_train, run_type='train'), get_embedding(args, files_val, run_type='val'), get_embedding(args, files_test, run_type='test')
    end = time.time()
    for ratio in train_ratios:
        args['train_ratio'] = ratio
        split_point = int(ratio*x_train.shape[0])

        report, mf1, params = classify(args, x_train[0:split_point], x_val, x_test, y_train[0:split_point], y_val, y_test, label_dict)
        save_report(args, params, report, mf1, end-start)

    args['method'] = 'seb'
    start = time.time()
    x_train, x_val, x_test = get_embedding(args, files_train, run_type='train'), get_embedding(args, files_val, run_type='val'), get_embedding(args, files_test, run_type='test')
    end = time.time()
    for ratio in train_ratios:
        args['train_ratio'] = ratio
        split_point = int(ratio*x_train.shape[0])

        report, mf1, params = classify(args, x_train[0:split_point], x_val, x_test, y_train[0:split_point], y_val, y_test, label_dict)
        save_report(args, params, report, mf1, end-start)

    args['method'] = 'lsd'
    start = time.time()
    x_train, x_val, x_test = get_embedding(args, files_train, run_type='train'), get_embedding(args, files_val, run_type='val'), get_embedding(args, files_test, run_type='test')
    end = time.time()
    for ratio in train_ratios:
        args['train_ratio'] = ratio
        split_point = int(ratio*x_train.shape[0])

        report, mf1, params = classify(args, x_train[0:split_point], x_val, x_test, y_train[0:split_point], y_val, y_test, label_dict)
        save_report(args, params, report, mf1, end-start)

    args['method'] = 'feather'
    start = time.time()
    x_train, x_val, x_test = get_embedding(args, files_train, run_type='train'), get_embedding(args, files_val, run_type='val'), get_embedding(args, files_test, run_type='test')
    end = time.time()
    for ratio in train_ratios:
        args['train_ratio'] = ratio
        split_point = int(ratio*x_train.shape[0])

        report, mf1, params = classify(args, x_train[0:split_point], x_val, x_test, y_train[0:split_point], y_val, y_test, label_dict)
        save_report(args, params, report, mf1, end-start)

    args['method'] = 'wl'
    for ratio in [1.0]:
        args['train_ratio'] = ratio
        split_point = int(ratio*len(files_train))

        start = time.time()
        x_train, x_val, x_test = get_kernel_embedding(args, files_train[0:split_point], files_val, files_test)
        end = time.time()
        report, mf1, params = classify(args, x_train, x_val, x_test, y_train[0:split_point], y_val, y_test, label_dict)
        save_report(args, params, report, mf1, end-start)


def main():
    from config import dm_args as args

    groups = ['type']

    for group in groups:
        args['group'] = group
        limit = np.inf

        print('\n\n************Group "{}"************\n\n'.format(args['group']))
        run_all_methods(args, limit)


if __name__ == '__main__':
    main()
