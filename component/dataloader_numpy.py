import pickle

import torch
from torch.utils.data import Dataset

from component.configuration import DATA_PATH, DEVICE


class CoughDataSet(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        self.data.loc[self.data['label'] >= 0.9, ['label']] = 1
        self.data.loc[self.data['label'] < 0.9, ['label']] = 0
        self.data = self.data.astype({'window_index': 'int64'})
        self.data = self.data.astype({'label': 'int32'})
        self.data = self.data.to_numpy()
        self.label = torch.from_numpy(self.data[:, 2]).to(DEVICE).long()
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.data[index]
        with open("%s/dnn2016_%d.pkl" % (DATA_PATH, row[0]), 'rb') as file_handler:
            data = pickle.load(file_handler)
            return torch.from_numpy(data['zxx_log'][:, row[1]: row[1] + 16]).to(DEVICE).double(), self.label.narrow(0,
                                                                                                                    index,
                                                                                                                    1).max()
