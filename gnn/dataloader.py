import torch
import numpy as np
from glob import glob

from torch_geometric.data import Dataset


class MalnetDataset(Dataset):
    def __init__(self, args, root, files, labels, transform=None, pre_transform=None):
        self.args = args
        self.files = files
        self.labels = labels
        self.num_classes = len(np.unique(labels))

        super(MalnetDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return glob(self.processed_dir.replace('/processed', '') + '/*.pt')

    def download(self):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.processed_dir.replace('/processed', '') + '/data_{}.pt'.format(idx))
        x.y = self.labels[idx]

        return x