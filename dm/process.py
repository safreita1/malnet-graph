def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import networkx as nx
from tqdm import tqdm
from grakel import graph_from_networkx
from joblib import Parallel, delayed


def process_file_karate(file):
    g = nx.read_edgelist(file)
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(gcc[0])
    g = nx.convert_node_labels_to_integers(g)

    return g


def process_file_nog(file):
    return nx.read_edgelist(file)


def process_file_slaq(file):
    g = nx.read_edgelist(file)
    g = nx.convert_node_labels_to_integers(g)
    adj = nx.to_scipy_sparse_matrix(g, dtype=np.float32, format='csr')
    adj.data = np.ones(adj.data.shape, dtype=np.float32)  # Set all elements to one in case of duplicate rows

    return adj


def process_file_grakel(file):
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
            delayed(process_file_grakel)(file)
            for file in files)

        data = gk.transform(graphs)
        if idx == 0:
            embedding = data
        else:
            embedding = np.concatenate((embedding, data), axis=0)
        pbar.update(chunk_size)
    pbar.close()

    return embedding