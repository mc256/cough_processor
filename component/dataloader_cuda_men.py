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

        if not row['audio'] in self.cache:
            with open("%s/dnn2016_%d.pkl" % (DATA_PATH, row['audio']), 'rb') as file_handler:
                data = pickle.load(file_handler)
                self.cache[row['audio']] = torch.from_numpy(data['zxx_log']).to(DEVICE).double()

        return self.cache[row['audio']].narrow_copy(1, row['window_index'].item(), 16), \
               torch.tensor(row['label'], dtype=torch.long, device=DEVICE)
