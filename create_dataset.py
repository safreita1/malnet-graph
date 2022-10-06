import os
import igraph as ig
import networkx as nx
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from androguard.misc import AnalyzeAPK


def visualize_graph(graph, path, og_set=None):
    g = ig.Graph(len(graph), list(zip(*list(zip(*nx.to_edgelist(graph)))[:2])), directed=True)
    layout = g.layout("kk")

    visual_style = {
        'vertex_size': 10,
        'vertex_color': '#AAAAFF',
        'edge_width': 1,
        'arrow_size': .01,
        'vertex_label': range(g.vcount()),
        'layout': layout
    }

    if og_set is not None:
        red_edges = g.es.select(_source_in=og_set, _target_in=og_set)
        red_edges["color"] = "red"

    ig.plot(g, path, **visual_style, bbox=(1000, 1000), margin=120, hovermode='closest')


def remap_graph(graph):
    node_index = 0
    node_mapping = {}

    remapped_graph = nx.DiGraph()

    for node in graph.nodes():
        node_mapping[node] = node_index
        node_index += 1

    for (n1, n2) in graph.edges():
        n1 = node_mapping[n1]
        n2 = node_mapping[n2]

        remapped_graph.add_edge(n1, n2)

    return remapped_graph


def create_fcg_graph(dex_path):
    save_path = dex_path.replace('apk_files', 'graph_files').replace('.apk', '.edgelist')

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            a, d, dx = AnalyzeAPK(dex_path)

            cg = dx.get_call_graph()
            cg_stripped = remap_graph(cg)
            visualize_graph(cg_stripped, path=save_path.replace('.edgelist', '.png'))

            if len(cg_stripped) > 0:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                with open(save_path, 'w') as file:
                    file.write('# Directed graph (each unordered pair of nodes is saved once)\n')
                    file.write('# Function call graph of malicious Android APK\n')
                    file.write('# SHA-256: {}\n'.format(save_path.rsplit('/', 1)[1].replace('.edgelist', '')))
                    file.write('# Nodes: {}, Edges: {}\n'.format(len(cg_stripped), len(list(cg_stripped.edges))))
                    file.write('# FromNodeId\tToNodeId\n')

                    for edge in cg_stripped.edges():
                        file.write('{}\t{}\n'.format(edge[0], edge[1]))

        except Exception as e:
            print('Error extracting FCG', e, dex_path)


def main():
    print('Constructing FCG Dataset')
    apk_files = glob(os.path.join(os.getcwd(), 'apk_files/*.apk'))

    Parallel(n_jobs=1)(
        delayed(create_fcg_graph)(apk_path)
        for apk_path in tqdm(apk_files))


if __name__ == '__main__':
    main()
