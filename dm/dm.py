import numpy as np
from tqdm import tqdm
from grakel import GraphKernel
from joblib import Parallel, delayed

from slaq import netlsd, netlsd_naive, vnge, vnge_naive
from karate import feather, fgsd, sf, ldp, geo_scattering, g2v_document
from process import process_file_karate, process_file_slaq, process_file_grakel, process_file_nog, kernel_transform


def run_method(idx, args, file, method):
    if method == 'sf':
        graph = process_file_karate(file)
        result = sf(graph, args['n_eigen'])

    elif method == 'ldp':
        subgraphs = process_file_karate(file)
        result = ldp(subgraphs)

    elif method == 'fgsd':
        graph = process_file_karate(file)
        result = fgsd(graph)

    elif method == 'feather':
        graph = process_file_karate(file)
        result = feather(graph, order=args['order'])

    elif method == 'geo_scattering':
        graph = process_file_karate(file)
        result = geo_scattering(graph, order=args['order'])

    elif method == 'g2v':
        graph = process_file_karate(file)
        result = g2v_document(idx, graph)

    elif method == 'lsd':
        graph = process_file_slaq(file)
        result = netlsd_naive(graph)

    elif method == 'lsd_slaq':
        graph = process_file_slaq(file)
        result = netlsd(graph, lanczos_steps=args['n_steps'], nvectors=args['n_vectors'])

    elif method == 'vnge_slaq':
        graph = process_file_slaq(file)
        result = vnge(graph, lanczos_steps=args['n_steps'], nvectors=args['n_vectors'])

    elif method == 'vnge':
        graph = process_file_slaq(file)
        result = vnge_naive(graph)

    elif method == 'nog':
        graph = process_file_nog(file)
        result = np.array([graph.number_of_nodes(), graph.number_of_edges()], dtype=np.int64)

    else:
        print('Method {} not implemented'.format(method))
        exit(1)

    return result


def get_kernel_embedding(args, train_files, val_files, test_files):
    print('\n******Running WL Kernel on train set******')
    gk = GraphKernel(kernel=[{'name': 'weisfeiler_lehman', 'n_iter': args['n_iter']}, 'subtree_wl'], normalize=True, n_jobs=args['n_cores'])

    graphs = Parallel(n_jobs=args['n_cores'])(
        delayed(process_file_grakel)(file)
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
        delayed(run_method)(idx, args, file, args['method'])
        for idx, file in enumerate(tqdm(files)))

    embedding = np.asarray(embedding)
    if len(embedding.shape) == 1: embedding = embedding.reshape(-1, 1)

    return embedding
