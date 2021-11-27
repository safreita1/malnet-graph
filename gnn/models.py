import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GINConv, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.args = args

        self.layers = torch.nn.ModuleList([])
        for i in range(args['num_layers'] + 1):
            dim_input = args['num_features'] if i == 0 else args['hidden_dim']

            nn = Sequential(Linear(dim_input, args['hidden_dim']), ReLU(), Linear(args['hidden_dim'], args['hidden_dim']))
            conv = GINConv(nn)

            self.layers.append(conv)

        self.fc1 = Linear(args['hidden_dim'], args['hidden_dim'])
        self.fc2 = Linear(args['hidden_dim'], args['num_classes'])

    def forward(self, x, edge_index, batch):
        for i, _ in enumerate(self.layers):
            x = F.relu(self.layers[i](x, edge_index))

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.args['dropout'], training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args

        self.layers = torch.nn.ModuleList([])
        for i in range(args['num_layers'] + 1):
            dim_input = args['num_features'] if i == 0 else args['hidden_dim']
            dim_output = args['num_classes'] if i == args['num_layers'] else args['hidden_dim']

            linear = Linear(dim_input, dim_output)
            self.layers.append(linear)

    def forward(self, x, edge_index, batch):
        x = global_add_pool(x, batch)

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
            else:  # last layer
                x = F.dropout(x, p=self.args['dropout'], training=self.training)
                x = layer(x)

        return F.log_softmax(x, dim=-1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.layers = torch.nn.ModuleList([])
        for i in range(args['num_layers'] + 1):
            dim_input = args['num_features'] if i == 0 else args['hidden_dim']

            conv = SAGEConv(dim_input, args['hidden_dim'])
            self.layers.append(conv)

        # for graph classification
        self.fc1 = torch.nn.Linear((args['num_layers'] + 1) * args['hidden_dim'], args['hidden_dim'])
        self.fc2 = torch.nn.Linear(args['hidden_dim'], args['num_classes'])

    def forward(self, x, edge_index, batch):
        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x_all.append(x)

        x = torch.cat(x_all, dim=1)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.args['dropout'], training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args

        self.layers = torch.nn.ModuleList([])
        for i in range(args['num_layers'] + 1):
            dim_input = args['num_features'] if i == 0 else args['hidden_dim']

            conv = GCNConv(dim_input, args['hidden_dim'])
            self.layers.append(conv)

        self.fc1 = Linear(args['hidden_dim'], args['hidden_dim'])
        self.fc2 = Linear(args['hidden_dim'], args['num_classes'])

    def forward(self, x, edge_index, batch):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x, edge_index))

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.args['dropout'], training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)


class SGC(torch.nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        self.args = args

        self.layers = torch.nn.ModuleList([])
        for i in range(args['num_layers'] + 1):
            dim_input = args['num_features'] if i == 0 else args['hidden_dim']

            conv = SGConv(dim_input, args['hidden_dim'], K=args['K'], add_self_loops=False, cached=False)
            self.layers.append(conv)

        self.fc1 = Linear(args['hidden_dim'], args['hidden_dim'])
        self.fc2 = Linear(args['hidden_dim'], args['num_classes'])

    def forward(self, x, edge_index, batch):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x, edge_index))

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.args['dropout'], training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
