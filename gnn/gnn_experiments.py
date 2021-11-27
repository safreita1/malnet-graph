import os
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed


def model_search(gpu, malnet_tiny, group, metric, epochs, model, K, num_layers, hidden_dim, lr, dropout, train_ratio):
    from config import args

    args.update({
        'gpu': gpu,
        'batch_size': 64,

        'node_feature': 'ldp',
        'directed_graph': True,
        'remove_isolates': False,
        'lcc_only': False,
        'add_self_loops': True,

        'model': model,
        'K': K,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,

        'metric': metric,
        'lr': lr,
        'dropout': dropout,
        'epochs': epochs,


        'group': group,
        'train_ratio': train_ratio,
        'malnet_tiny': malnet_tiny


    })

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])

    from gnn import run_experiment
    val_score, test_score, param_count, run_time = run_experiment(args)
    return args, val_score, test_score, param_count, run_time


def preprocess_search(gpu, epochs, node_feature, directed_graph, remove_isolates, lcc_only, add_self_loops, model='gcn', K=0, hidden_dim=32, num_layers=3, lr=0.0001, dropout=0):
    from config import args

    args.update({
        'gpu': gpu,
        'batch_size': 128,

        'node_feature': node_feature,
        'directed_graph': directed_graph,
        'remove_isolates': remove_isolates,
        'lcc_only': lcc_only,
        'add_self_loops': add_self_loops,

        'model': model,
        'K': K,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,

        'lr': lr,
        'dropout': dropout,
        'epochs': epochs,

        'group': 'type',
        'train_ratio': 1.0,
        'malnet_tiny': True
    })

    from gnn import run_experiment
    val_score, test_score, param_count, run_time = run_experiment(args, args['group'], gpu)
    return args, val_score, test_score, param_count, run_time


def search_all_preprocess():
    epochs = 1000
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]

    # Test node features
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature=feature, directed_graph=True, remove_isolates=True, lcc_only=False, add_self_loops=False)
        for idx, feature in enumerate(tqdm(['ldp', 'constant', 'degree'])))

    # Test directed graph
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=directed, remove_isolates=True, lcc_only=False, add_self_loops=False)
        for idx, directed in enumerate(tqdm([True, False])))

    # Test isolates
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=True, remove_isolates=isolates, lcc_only=False, add_self_loops=False)
        for idx, isolates in enumerate(tqdm([True, False])))

    # Test lcc
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=False, remove_isolates=True, lcc_only=lcc, add_self_loops=False)
        for idx, lcc in enumerate(tqdm([True, False])))

    # Test self loops
    Parallel(n_jobs=len(gpus))(
        delayed(preprocess_search)(gpus[idx], epochs, node_feature='constant', directed_graph=True, remove_isolates=True, lcc_only=False, add_self_loops=self_loops)
        for idx, self_loops in enumerate(tqdm([True, False])))


def search_all_models():
    gpus = [2]

    models = ['gin']
    layers = [5]
    hidden_dims = [64]
    learning_rates = [0.0001]
    dropouts = [0]
    epochs = 500
    metric = 'macro-F1'
    groups = ['family']  # , 'family'
    malnet_tiny = False
    train_ratios = [1.0]  # , 0.01, 0.001

    # Search for GCN, GraphSage, GIN
    combinations = list(itertools.product(*[groups, models, layers, hidden_dims, learning_rates, dropouts, train_ratios]))

    results = Parallel(n_jobs=len(combinations))(
        delayed(model_search)(gpus[idx % len(gpus)], malnet_tiny, group, metric, epochs, model=model, K=0, num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, dropout=dropout, train_ratio=train_ratio)
        for idx, (group, model, num_layers, hidden_dim, lr, dropout, train_ratio) in enumerate(tqdm(combinations)))

    # Search for SGC
    # models, K = ['sgc'], [1, 2, 3]
    # combinations = list(itertools.product(*[models, layers, hidden_dims, learning_rates, dropouts, K]))
    #
    # results_sgc = Parallel(n_jobs=len(combinations))(
    #     delayed(model_search)(gpus[idx % len(gpus)], malnet_tiny, group, metric, epochs, model=model, K=K, num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, dropout=dropout)
    #     for idx, (group, model, num_layers, hidden_dim, lr, dropout, K) in enumerate(tqdm(combinations)))
    #
    # # Search for MLP
    # models, layers = ['mlp'], [1, 3, 5]
    # combinations = list(itertools.product(*[models, layers, hidden_dims, learning_rates, dropouts]))
    #
    # results_mlp = Parallel(n_jobs=len(combinations))(
    #     delayed(model_search)(gpus[idx % len(gpus)], malnet_tiny, group, metric, epochs, model=model, K=0, num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, dropout=dropout)
    #     for idx, (group, model, num_layers, hidden_dim, lr, dropout) in enumerate(tqdm(combinations)))
    #
    # results = results_gin_gcn + results_sgc + results_mlp

    for (args, val_score, test_score, param_count, run_time) in results:
        print('Tiny={}, group={}, train_ratio={}, model={}, epochs={}, run time={} seconds, # parameters={}, layers={}, hidden_dims={}, learning_rate={}, dropout={}, val_score={}, test_score={}'.format(
            args['malnet_tiny'], args['group'], args['train_ratio'], args['model'], args['epochs'], run_time, param_count, args['num_layers'], args['hidden_dim'], args['lr'], args['dropout'], val_score, test_score))


def run_best_models():
    epochs = 500
    gpus = [2, 3, 4, 5]
    metric = 'macro-F1'
    group = 'family'
    malnet_tiny = True

    # model, K, layers, hidden_dim, learning_rate, dropout
    combinations = [['gin', 0, 3, 64, 0.001, 0.5]]  # ['sgc', 1, 3, 64, 0.001, 0.5], ['gcn', 0, 5, 64, 0.001, 0.5], ['mlp', 0, 1, 128, 0.001, 0], ['graphsage', 0, 5, 128, 0.0001, 0]

    results = Parallel(n_jobs=len(combinations))(
        delayed(model_search)(gpus[idx % len(gpus)], malnet_tiny, group, metric, epochs, model=model, K=K, num_layers=num_layers, hidden_dim=hidden_dim, lr=lr, dropout=dropout)
        for idx, (model, K, num_layers, hidden_dim, lr, dropout) in enumerate(tqdm(combinations)))

    for (args, val_score, test_score, param_count, run_time) in results:
            print('Tiny={}, group={}, train_ratio={}, model={}, epochs={}, run time={} seconds, # parameters={}, layers={}, hidden_dims={}, learning_rate={}, dropout={}, val_score={}, test_score={}'.format(
                args['malnet_tiny'], args['group'], args['train_ratio'], args['model'], args['epochs'], run_time, param_count, args['num_layers'], args['hidden_dim'], args['lr'], args['dropout'], val_score, test_score))


if __name__ == '__main__':
    # search_all_preprocess()
    search_all_models()
    # run_best_models()
