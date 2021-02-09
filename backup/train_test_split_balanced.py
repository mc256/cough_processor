# %%

import os
import torch
import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import pickle
from tqdm import tqdm as tqdm

# %%
SEED = 2333333
WINDOW_SIZE = 64
SR = 16000

# %%
# Base Path
BASE_PATH = '/home/jlchen/sandbox/'
TRAIN_LABEL_PATH = './label/segmented_train_set.csv'
VAL_LABEL_PATH = './label/segmented_val_set.csv'


# %%
def balance_data_set(label_path, ratio=0.5, positive_threshold=0.9):
    table_labels = pd.read_csv(label_path, index_col=0)
    table_labels.loc[table_labels['label'] >= 0.9, ['label']] = 1
    positive_count = table_labels[table_labels['label'] == 1].shape[0]

    positive_samples = table_labels[table_labels['label'] == 1].sample(n=positive_count, random_state=SEED)
    negative_samples = table_labels[table_labels['label'] == 0].sample(n=positive_count, random_state=SEED)

    all_samples = pd.concat((positive_samples, negative_samples), ignore_index=True)
    print(all_samples.describe())

    return all_samples


balance_data_set(TRAIN_LABEL_PATH, 0.5).to_csv('./label/segmented_train_set_balanced.csv')
balance_data_set(VAL_LABEL_PATH, 0.5).to_csv('./label/segmented_val_set_balanced.csv')





