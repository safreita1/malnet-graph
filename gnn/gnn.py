import os
import sys
import copy
import time
import torch
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, '..')
from utils import get_split_info
from dataloader import MalnetDataset
from models import GIN, GraphSAGE, MLP, GCN, SGC
from process import NodeDegree, save_model, log_info, convert_files_pytorch


gnn_models = {
    'gin': GIN,
    'graphsage': GraphSAGE,
    'mlp': MLP,
    'gcn': GCN,
    'sgc': SGC
}

node_features = {
    'ldp': T.LocalDegreeProfile(),
    'constant': T.Constant(),
    'degree': NodeDegree()
}


def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def train(model, device, optimizer, train_loader, train_dataset, epoch):
    model.train()

    loss_all = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)


def test(model, device, loader):
    model.eval()

    y_true, y_pred, y_scores = [], [], []
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]

        y_true.extend(data.y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        y_scores.extend(output[:, 1].tolist())  # only used in binary setting

    return y_pred, y_scores, y_true


def train_model(args, device, train_dataset, train_loader, val_loader, test_loader):
    model = gnn_models[args['model']](args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    writer = SummaryWriter(log_dir=args['log_dir'])

    best_val_score = 0
    for epoch in range(1, args['epochs'] + 1):

        start = time.time()
        train_loss = train(model, device, optimizer, train_loader, train_dataset, epoch)
        end = time.time()

        y_pred_val, y_scores_val, y_true_val = test(model, device, val_loader)
        y_pred_test, y_scores_test, y_true_test = test(model, device, test_loader)

        val_score = accuracy_score(y_true_val, y_pred_val) if args['metric'] == 'acc' else f1_score(y_true_val, y_pred_val, average='macro')
        test_score = accuracy_score(y_true_test, y_pred_test) if args['metric'] == 'acc' else f1_score(y_true_test, y_pred_test, average='macro')

        writer.add_scalars(main_tag='Tiny={}, train_ratio={}, group={} model={}, layers={}, hidden_dims={}, learning_rate={}, dropout={}'.format(
                args['malnet_tiny'], args['train_ratio'], args['group'], args['model'], args['num_layers'], args['hidden_dim'], args['lr'], args['dropout']),
            global_step=epoch,
            tag_scalar_dict={'Validation {}'.format(args['metric']): val_score, 'Test {}'.format(args['metric']): test_score}
        )

        with open(args['log_dir'] + 'train_results.txt', 'a') as f:
            f.write('Tiny={}, group={}, train_ratio={} Epoch={}, time={} seconds, model={}, # parameters={}, layers={}, hidden_dims={}, learning_rate={}, dropout={}, train_loss={}, val_{}={}, test_{}={}\n'.format(
                args['malnet_tiny'], args['group'], args['train_ratio'], epoch, round(end-start, 2), args['model'], get_parameter_count(model), args['num_layers'], args['hidden_dim'], args['lr'], args['dropout'], train_loss, args['metric'], val_score, args['metric'], test_score))

        if not args['quiet']: print('Epoch: {:03d}, Train Loss: {:.7f}, Val {}: {:.7f}'.format(epoch, train_loss, args['metric'], val_score))

        if val_score > best_val_score:
            if not args['quiet']: print('Improved val {} from {} to {} at epoch {}. Saving and logging model.'.format(args['metric'], best_val_score, val_score, epoch))
            best_val_score = val_score

            save_model(args, model)
            log_info(args, epoch, y_true_val, y_pred_val, y_scores_val, param_count=0, run_time=0, data_type='val')

    print('Best val {}: {}'.format(args['metric'], best_val_score))

    # load best model
    model.load_state_dict(torch.load(args['log_dir'] + 'best_model.pt'))
    model.eval()

    return model


def run_experiment(args_og):
    args = copy.deepcopy(args_og)

    if args['model'] != 'sgc': args['K'] = 0

    args['log_dir'] = os.getcwd() + '/results/malnet_tiny={}/group={}/train_ratio={}/node_feature={}/directed_graph={}' \
                                    '/remove_isolates={}/lcc_only={}/add_self_loops={}/model={}/K={}/hidden_dim={}' \
                                    '/num_layers={}/lr={}/dropout={}/epochs={}/'.format(args['malnet_tiny'], args['group'],
                                                        args['train_ratio'], args['node_feature'], args['directed_graph'],
                                                        args['remove_isolates'], args['lcc_only'], args['add_self_loops'],
                                                        args['model'], args['K'], args['hidden_dim'], args['num_layers'],
                                                        args['lr'], args['dropout'], args['epochs'])

    args['data_dir'] = os.getcwd() + '/data/malnet_tiny={}/group={}/train_ratio={}/node_feature={}/directed_graph={}' \
                                     '/remove_isolates={}/lcc_only={}/add_self_loops={}/'.format(args['malnet_tiny'], args['group'],
                                                        args['train_ratio'], args['node_feature'], args['directed_graph'],
                                                        args['remove_isolates'], args['lcc_only'], args['add_self_loops'])
    os.makedirs((args['log_dir']), exist_ok=True)

    train_dir = args['data_dir'].replace('/data/', '/data/train/')
    val_dir = args['data_dir'].replace('/data/', '/data/val/').replace('/train_ratio={}'.format(args['train_ratio']), '/train_ratio=1.0')
    test_dir = args['data_dir'].replace('/data/', '/data/test/').replace('/train_ratio={}'.format(args['train_ratio']), '/train_ratio=1.0')

    files_train, files_val, files_test, train_labels, val_labels, test_labels, label_dict = get_split_info(args)
    convert_files_pytorch(args, files_train, train_dir, node_features[args['node_feature']])
    convert_files_pytorch(args, files_val, val_dir, node_features[args['node_feature']])
    convert_files_pytorch(args, files_test, test_dir, node_features[args['node_feature']])

    train_dataset = MalnetDataset(args, root=train_dir, files=files_train, labels=train_labels)
    val_dataset = MalnetDataset(args, root=val_dir, files=files_val, labels=val_labels)
    test_dataset = MalnetDataset(args, root=test_dir, files=files_test, labels=test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'])

    args['num_classes'] = train_dataset.num_classes
    args['num_features'] = train_dataset.num_features
    args['class_indexes'] = list(label_dict.values())
    args['class_labels'] = list(label_dict.keys())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    model = train_model(args, device, train_dataset, train_loader, val_loader, test_loader)
    run_time = round(time.time() - start, 2)

    param_count = get_parameter_count(model)
    y_pred_val, y_scores_val, y_true_val = test(model, device, val_loader)
    y_pred_test, y_scores_test, y_true_test = test(model, device, test_loader)
    log_info(args, args['epochs'], y_true_test, y_pred_test, y_scores_test, param_count, run_time=run_time, data_type='test')

    val_score = accuracy_score(y_true_val, y_pred_val) if args['metric'] == 'acc' else f1_score(y_true_val, y_pred_val, average='macro')
    test_score = accuracy_score(y_true_test, y_pred_test) if args['metric'] == 'acc' else f1_score(y_true_test, y_pred_test, average='macro')

    return val_score, test_score, param_count, run_time


def main():
    from config import args

    gpus = [0]
    groups = ['type']

    Parallel(n_jobs=len(groups))(
        delayed(run_experiment)(args, group, gpus[idx])
        for idx, group in enumerate(tqdm(groups)))


if __name__ == '__main__':
    main()
