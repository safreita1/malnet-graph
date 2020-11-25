def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
from glob import glob
from grakel import graph_from_networkx
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from seb import _get_subgraphs, _function_basis


def process_file_for_karate(file):
    g = nx.read_edgelist(file)
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(gcc[0])
    g = nx.convert_node_labels_to_integers(g)

    return g


def process_file_for_nog(file):
    return nx.read_edgelist(file)


def process_file_for_seb(file, norm_flag='yes'):
    g = nx.read_edgelist(file)
    g.remove_edges_from(nx.selfloop_edges(g))
    g = nx.convert_node_labels_to_integers(g)

    subgraphs = _get_subgraphs(g)
    subgraphs = [_function_basis(gi, ['deg'], norm_flag=norm_flag) for gi in subgraphs]
    subgraphs = [g for g in subgraphs if g is not None]

    return subgraphs


def process_file_for_slaq(file):
    g = nx.read_edgelist(file)
    g = nx.convert_node_labels_to_integers(g)
    adj = nx.to_scipy_sparse_matrix(g, dtype=np.float32, format='csr')
    adj.data = np.ones(adj.data.shape, dtype=np.float32)  # Set all elements to one in case of duplicate rows

    return adj


def process_file_for_grakel(file):
    g = nx.read_edgelist(file)
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(gcc[0])
    g = nx.convert_node_labels_to_integers(g)
    nx.set_node_attributes(g, 'a', 'label')

    return list(graph_from_networkx([g], node_labels_tag='label', as_Graph=True))[0]


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def kernel_transform(args, files, gk):
    chunk_size = 1000
    pbar = tqdm(total=len(files))

    for idx, files in enumerate(chunker(files, chunk_size)):
        graphs = Parallel(n_jobs=args['n_cores'])(
            delayed(process_file_for_grakel)(file)
            for file in files)

        data = gk.transform(graphs)
        if idx == 0:
            embedding = data
        else:
            embedding = np.concatenate((embedding, data), axis=0)
        pbar.update(chunk_size)
    pbar.close()

    return embedding


def split_rand(labels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=1, shuffle=True):
    train_idx, test_idx = train_test_split(np.arange(len(labels)), train_size=train_ratio * 0.7, test_size=val_ratio + test_ratio, random_state=seed, shuffle=shuffle, stratify=labels)
    val_idx, test_idx = train_test_split(test_idx, train_size=val_ratio/(val_ratio+test_ratio), test_size=test_ratio/(val_ratio+test_ratio), random_state=seed, shuffle=shuffle, stratify=np.asarray(labels)[test_idx])

    print('Number of train samples: {}, val samples: {}, test samples: {}'.format(len(train_idx), len(val_idx), len(test_idx)))
    return train_idx, val_idx, test_idx


def save_split_info(args, files, train_idx, val_idx, test_idx):
    def save_split(args, files, split):
        save_dir = args['results_dir'] + '/split_info/{}/{}/'.format(args['group'], args['train_ratio'])
        os.makedirs(save_dir, exist_ok=True)

        with open(save_dir + '{}.txt'.format(split), 'w') as f:
            for file in files:
                file = file.split('/graph_data/')[1].rsplit('.')[0]
                f.write(file + '\n')

    train_files = np.asarray(files)[train_idx].tolist()
    val_files = np.asarray(files)[val_idx].tolist()
    test_files = np.asarray(files)[test_idx].tolist()

    save_split(args, train_files, split='train')
    save_split(args, val_files, split='val')
    save_split(args, test_files, split='test')


def get_data(args, limit=np.inf):
    files = []
    labels = []
    label_idx = 0
    label_dict = {}

    print('Loading graph info...')
    if args['group'] == 'type':
        mtype_dirs = glob(args['graph_dir'] + '*/')
        mtype_dirs = [mtype_dir for mtype_dir in mtype_dirs if '*' not in mtype_dir]
        m_paths = defaultdict(list)

        for mtype_dir in mtype_dirs:
            mtype = mtype_dir.split('/graph_data/')[1].split('/')[0]
            m_paths[mtype] = glob(mtype_dir + '/*/*.edgelist')

    elif args['group'] == 'family':
        mfam_dirs = glob(args['graph_dir'] + '*/*/')
        mfam_dirs = [mtype_dir for mtype_dir in mfam_dirs if '*' not in mtype_dir]
        m_paths = defaultdict(list)

        for mfam_dir in mfam_dirs:
            mfam = mfam_dir.split('/graph_data/')[1].split('/')[1]
            m_paths[mfam].extend(glob(mfam_dir + '/*.edgelist'))
    else:
        print('Group not valid')
        exit(1)

    for mal_group, paths in m_paths.items():
        if len(paths) < limit:
            files.extend(paths)
            labels.extend([label_idx] * len(paths))
            label_dict[mal_group] = label_idx
            label_idx += 1

    files, labels = (list(t) for t in zip(*sorted(zip(files, labels))))

    print('Finished loading graph info')
    return files, labels, label_dict
