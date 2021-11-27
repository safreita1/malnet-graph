import os
import torch
import networkx as nx
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import torch_geometric.transforms as T
from torch_geometric.utils.convert import from_networkx
from pprint import pprint
from torch_geometric.utils import degree
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score


def process_file(args, idx, file, processed_dir, pre_transform):
    if args['directed_graph']:
        graph = nx.read_edgelist(file, create_using=nx.DiGraph)
    else:
        graph = nx.read_edgelist(file)

    if args['lcc_only']:
        graph = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])

    # networkx --> pytorch geometric
    data = from_networkx(graph)

    if args['remove_isolates']:
        data = T.RemoveIsolatedNodes()(data)

    if args['add_self_loops']:
        data = T.AddSelfLoops()(data)

    if pre_transform is not None:
        data = pre_transform(data)

    torch.save(data, processed_dir + 'data_{}.pt'.format(idx))


def convert_files_pytorch(args, files, processed_dir, pre_transform):
    # check if processed files exist
    if len(glob(processed_dir + '*.pt')) != len(files):
        os.makedirs(processed_dir, exist_ok=True)

        Parallel(n_jobs=args['n_cores'])(
            delayed(process_file)(args, idx, file, processed_dir, pre_transform)
            for idx, file in enumerate(tqdm(files)))


class NodeDegree(object):
    def __call__(self, data):
        row, col = data.edge_index
        N = data.num_nodes

        data.x = degree(row, N, dtype=torch.float)
        data.x = data.x.view(-1, 1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def save_model(args, model):
    torch.save(model.state_dict(), args['log_dir'] + 'best_model.pt')


def log_info(args, epoch, y_true, y_pred, y_scores, param_count, run_time, data_type='val'):
    macro_f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
    report = classification_report(y_true, y_pred, labels=args['class_indexes'], target_names=args['class_labels'], output_dict=True)

    with open(args['log_dir'] + 'best_{}_info.txt'.format(data_type), 'w') as f:
        pprint('Parameters', stream=f)
        pprint(args, stream=f)
        pprint('Epoch: {}'.format(epoch), stream=f)
        pprint('Number of model parameters: {}'.format(param_count), stream=f)

        pprint('Classification report', stream=f)
        pprint(report, stream=f)
        pprint('Macro-f1: {}'.format(macro_f1), stream=f)
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
