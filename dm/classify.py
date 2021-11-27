import os
import gc
import sys
import itertools
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score, accuracy_score


def log_info(args, params, y_true, y_pred, y_scores, run_time):
    score = accuracy_score(y_true, y_pred) if args['metric'] == 'acc' else f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, labels=args['class_indexes'], target_names=args['class_labels'], output_dict=True)

    save_path = os.getcwd() + '/results/group={}/train_ratio={}/method={}/iter={}/order={}/n_eigen={}/n_vec={}/n_steps={}/info.txt'.format(
        args['group'], args['train_ratio'], args['method'], args['n_iter'], args['order'], args['n_eigen'], args['n_vectors'], args['n_steps'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        pprint('Parameters', stream=f)
        pprint(args, stream=f)
        pprint(params, stream=f)

        pprint('Classification report', stream=f)
        pprint(report, stream=f)
        pprint('{}: {}'.format(args['metric'], score), stream=f)
        pprint('Malware group: {}'.format(args['group']), stream=f)
        pprint('Train ratio: {}'.format(args['train_ratio']), stream=f)
        pprint('Embedding took {} seconds w/ {} cpu cores'.format(round(run_time, 2), args['n_cores']), stream=f)

        pprint('Label dictionary:', stream=f)
        pprint(args['class_labels'], stream=f)

        if args['group'] != 'family':
            pprint('Confusion matrix', stream=f)

            cm = confusion_matrix(y_true, y_pred, labels=args['class_indexes'])
            pprint(cm, stream=f)

        if args['group'] == 'binary':
            pprint('FPR/TPR Info', stream=f)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=[1])
            pprint('tpr: {}'.format(tpr.tolist()), stream=f)
            pprint('fpr: {}'.format(fpr.tolist()), stream=f)
            pprint('thresholds: {}'.format(thresholds.tolist()), stream=f)

            auc_macro_score = roc_auc_score(y_true, y_scores, average='macro')
            auc_class_scores = roc_auc_score(y_true, y_scores, average=None)
            pprint('AUC macro score: {}'.format(auc_macro_score), stream=f)
            pprint('AUC class scores: {}'.format(auc_class_scores), stream=f)


def grid_search(args, x_train, y_train, x_val, y_val):
    best_score = 0

    n_estimators = [1, 5, 10, 50]
    max_depths = [1, 5, 10, 20]
    params = list(itertools.product(n_estimators, max_depths))

    for n_estimator, max_depth in tqdm(params):
        clf = RandomForestClassifier(random_state=args['seed'], n_estimators=n_estimator, max_depth=max_depth, n_jobs=args['n_cores'])
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_val)
        score = f1_score(y_val, y_pred, average='macro')

        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n_estimator, 'max_depth': max_depth}

        del clf
        gc.collect()

    print('\nBest val {}: {}'.format(args['metric'], best_score))

    return best_params, best_score


def classify(args, x_train, x_val, x_test, y_train, y_val, y_test, run_time):
    print('\nClassifying {} graph embeddings...'.format(args['method']))

    # standardize the data
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_val, x_test = scalar.transform(x_val), scalar.transform(x_test)

    params, val_score = grid_search(args, x_train, y_train, x_val, y_val)

    clf = RandomForestClassifier(random_state=args['seed'], n_jobs=args['n_cores'])
    clf.set_params(**params)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_scores = clf.predict_proba(x_test)[:, 1].tolist()  # only used in binary setting

    log_info(args, params, y_test, y_pred, y_scores, run_time)
    test_score = accuracy_score(y_test, y_pred) if args['metric'] == 'acc' else f1_score(y_test, y_pred, average='macro')

    return val_score, test_score