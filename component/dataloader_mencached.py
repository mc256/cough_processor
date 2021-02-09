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
        self.length = len(data)
        self.cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.data.iloc[index]

        zxx_log = self.cache.get(row['audio'], False)
        if type(zxx_log) is bool:
            with open("%s/dnn2016_%d.pkl" % (DATA_PATH, row['audio']), 'rb') as file_handler:
                data = pickle.load(file_handler)
                self.cache[row['audio']] = data['zxx_log']
                zxx_log = data['zxx_log']

        return torch.from_numpy(zxx_log[:, row['window_index']: row['window_index'] + 16]) \
                   .to(DEVICE).double(), \
               torch.tensor(row['label'], dtype=torch.long, device=DEVICE)
