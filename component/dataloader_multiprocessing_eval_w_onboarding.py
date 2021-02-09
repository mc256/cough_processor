import pickle

import torch
from torch.utils.data import Dataset

from component.configuration import ONBOARDING_DATA_PATH


class CoughDataSet(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        self.data.loc[self.data['label'] >= 0.9, ['label']] = 1
        self.data.loc[self.data['label'] < 0.9, ['label']] = 0
        self.data = self.data.astype({'window_index': 'int64'})
        self.data = self.data.astype({'label': 'int32'})
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with open("%s/dnn2016_%d.pkl" % (ONBOARDING_DATA_PATH, row['audio']), 'rb') as file_handler:
            data = pickle.load(file_handler)
            return data['zxx_log'][:, row['window_index']: row['window_index'] + 16], row['label']
