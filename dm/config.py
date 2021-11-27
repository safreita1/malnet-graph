import multiprocessing

args = {
    'method': 'vnge',
    'metric': 'acc',  # options: 'acc', 'macro-f1'

    'n_iter': 5,  # used in WL
    'order': 4,  # used in Feather and GeoScattering
    'n_eigen': 100,  # used in SF
    'n_vectors': 10,  # used in SLAQ
    'n_steps': 10,  # used in SLAQ

    'group': 'family',  # options: 'family', 'type'. MalNet-Tiny only works with 'type'
    'train_ratio': 1.0,
    'val_ratio': 0.1,
    'test_ratio': 0.2,

    'seed': 0,
    'shuffle': True,
    'n_cores': multiprocessing.cpu_count(),

    'malnet_tiny': True,  # True = use MalNet-Tiny; False = use Malnet
    'malnet_dir': '/raid/sfreitas3/malnet-graphs/',  # ** USER SPECIFIED DIRECTORY TO MALNET DATA **
    'malnet_tiny_dir': '/raid/sfreitas3/malnet-graphs-tiny/'  # ** USER SPECIFIED DIRECTORY TO MALNET-TINY DATA **
}


