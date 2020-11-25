"""
Code authored Chen Cai: https://github.com/Chen-Cai-OSU/LDP

This a modified version of the implementation of the paper "A simple yet effective baseline for non-attribute graph classification", accepted by ICLR 2019 workshop on Representation learning on graphs and manifolds, for use with MalNet.

"""

import numpy as np
import networkx as nx
from statsmodels.distributions.empirical_distribution import ECDF


def _attribute_mean(g, i, key='deg', cutoff=1, iteration=0):
    for itr in [iteration]:
        assert key in g.nodes[i].keys()
        nodes = g[i].keys()

        if iteration == 0:
            nbrs_deg = [g.nodes[j][key] for j in nodes]
        else:
            key_ = str(cutoff) + '_' + str(itr-1) + '_' + key +  '_' + 'mean'
            nbrs_deg = [g.nodes[j][key_] for j in nodes]
            g.nodes[i][ str(cutoff) + '_' + str(itr) + '_' + key] = np.mean(nbrs_deg)
            return

        oldkey = key
        key = str(cutoff) + '_' + str(itr) + '_' + oldkey
        key_mean = key + '_mean'; key_min = key + '_min'; key_max = key + '_max'; key_std = key + '_std'
        key_sum = key + '_sum'

        if len(nbrs_deg) == 0:
            g.nodes[i][key_mean] = 0
            g.nodes[i][key_min] = 0
            g.nodes[i][key_max] = 0
            g.nodes[i][key_std] = 0
            g.nodes[i][key_sum] = 0
        else:
            # assert np.max(nbrs_deg) < 1.1
            g.nodes[i][key_mean] = np.mean(nbrs_deg)
            g.nodes[i][key_min] = np.min(nbrs_deg)
            g.nodes[i][key_max] = np.max(nbrs_deg)
            g.nodes[i][key_std] = np.std(nbrs_deg)
            g.nodes[i][key_sum] = np.sum(nbrs_deg)


def _function_basis(g, allowed, norm_flag='no'):
    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    # allowed = ['ricci', 'deg', 'hop', 'cc', 'fiedler']
    # to save recomputation. Look at the existing feature at first and then simply compute the new one.

    if len(g)<3: return
    assert nx.is_connected(g)

    def norm(g, key, flag=norm_flag):
        if flag=='no':
            return 1
        elif flag == 'yes':
            return np.max(np.abs(list(nx.get_node_attributes(g, key).values()))) + 1e-6

    if 'deg' in allowed:
        deg_dict = dict(nx.degree(g))
        for n in g.nodes():
            g.nodes[n]['deg'] = deg_dict[n]

        deg_norm = norm(g, 'deg', norm_flag)
        for n in g.nodes():
            g.nodes[n]['deg'] /= np.float(deg_norm)
    if 'deg' in allowed:
        for n in g.nodes():
            _attribute_mean(g, n, key='deg', cutoff=1, iteration=0)
        if norm_flag == 'yes':
            # better normalization
            for attr in ['1_0_deg_sum']:  # used to include 1_0_deg_std/ deleted now:
                norm_ = norm(g, attr, norm_flag)
                for n in g.nodes():
                    g.nodes[n][attr] /= float(norm_)
    return g


def _get_subgraphs(g, threshold=1):
    assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>"
    subgraphs = [g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    subgraphs = [c for c in subgraphs if len(c) > threshold]
    return subgraphs


def _functionongraph(subgraphs, key='deg', edge_flag=False):
    components = len(subgraphs)
    lis = []
    for j in range(components):
        g = subgraphs[j]
        try:
            assert (str(type(g)) == "<class 'networkx.classes.graphviews.SubGraph'>") or (str(type(g))) == "<class 'networkx.classes.graph.Graph'>"
        except AssertionError:
            if g is None:
                print('wired case: g is None')
                return [0]
            else:
                print('Unconsidered Cases in function on graph')

        if edge_flag == False:
            tmp = [g.nodes[k][key] for k in g.nodes]
        lis += tmp
    return lis


def _hisgram(lis, n_bin=100, his_norm_flag='yes', lowerbound=-1, upperbound=1, cdf_flag=False, uniform_flag=True):
    if lis == []:
        print('lis is empty')
        return [0]*n_bin

    if his_norm_flag == 'yes':
        try:
            assert max(lis) < 1.1
        except AssertionError:
            print('The max of list is %s' % max(lis)),
        assert min(lis) > -1.1

    if not uniform_flag:
        assert lowerbound + 1e-3 > 0
        n_bin_ = np.logspace(np.log(lowerbound + 1e-3), np.log(upperbound),n_bin+1, base = np.e)
    else:
        n_bin_ = n_bin

    if cdf_flag:
        ecdf = ECDF(lis)
        if uniform_flag:
            return ecdf([i / np.float(n_bin) for i in range(0, n_bin)])
        else:
            return ecdf([i / np.float(n_bin) for i in range(0, n_bin)])
    result = np.histogram(lis, bins=n_bin_, range=(lowerbound,upperbound))
    return result[0]


def seb(subgraphs, n_bin=50, key='deg', his_norm_flag='yes', edge_flag=False, lowerbound=-1, upperbound=1, cdf_flag=False, uniform_flag=True):
    lis = _functionongraph(subgraphs, key, edge_flag=edge_flag)
    embedding = _hisgram(lis, n_bin, his_norm_flag=his_norm_flag, lowerbound=lowerbound, upperbound=upperbound,  cdf_flag=cdf_flag, uniform_flag=uniform_flag)

    return embedding
