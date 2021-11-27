import multiprocessing

args = {
    # device
    'seed': 1,
    'gpu': 0,
    'n_cores': multiprocessing.cpu_count(),
    'batch_size': 256,
    'quiet': False,

    # pre-processing
    'node_feature': 'ldp',  # 'ldp', 'constant', 'degree'
    'directed_graph': True,
    'remove_isolates': False,
    'lcc_only': False,
    'add_self_loops': True,

    # net
    'model': 'gcn',  # 'sgc', 'gin', 'gcn', 'mlp'
    'K': 1,  # only for 'sgc' model
    'hidden_dim': 64,
    'num_layers': 3,

    # learning
    'metric': 'acc',  # options: 'acc', 'macro-f1'
    'lr': 0.001,
    'dropout': 0.5,
    'epochs': 5,

    # data
    'group': 'family',  # options: 'family', 'type'. MalNet-Tiny only works with 'type'
    'train_ratio': 1.0,  # corresponds to 'split_info' folder; controls percentage of training data
    'malnet_tiny': False,  # True = use MalNet-Tiny; False = use Malnet
    'malnet_dir': '/raid/sfreitas3/malnet-graphs/',  # ** USER SPECIFIED DIRECTORY TO MALNET DATA **
    'malnet_tiny_dir': '/raid/sfreitas3/malnet-graphs-tiny/'  # ** USER SPECIFIED DIRECTORY TO MALNET-TINY DATA **
}
