import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tqdm import tqdm

from component.cnn_model import ModelC
from component.dataloader import CoughDataSet

BATCH_SIZE = 20
INPUT_PIXEL_WIDTH = 8
SEED = 2333333

DEVICE = 'cuda:0'
#DEVICE = 'cpu'
#RUN = 'Dec19-balanced-30dB'
RUN = 'Dec19-30dB-ratio9'
MODEL = 49

DATA_PATH = '/home/jlchen/sandbox/feature/dnn_paper'
TEST_LABEL = './label/test_set.csv'

THRESHOLD = 1

black_box = ModelC().to(DEVICE).double()
print(black_box)


black_box.load_state_dict(torch.load('./model/cnn_attempt_%s_epoch_%d.pkl' % (RUN, MODEL)))
black_box.eval()

test_df = pd.read_csv('./label/segmented_val_set.csv')
test_positive = test_df[test_df['label'] >= 0.9]
test_negative = test_df[test_df['label'] < 0.9]
count_positive = len(test_positive)
count_negative = len(test_negative)
print(count_negative, count_positive)

positive_index = np.arange(0, count_positive, 1)
negative_index = np.arange(0, count_negative, 1)
np.random.shuffle(positive_index)
np.random.shuffle(negative_index)
TEST_CASES = 10000

test_sample = pd.concat(
    [
        test_positive.iloc[positive_index[0:TEST_CASES // 2]],
        test_negative.iloc[negative_index[0:TEST_CASES // 2]]
    ],
    ignore_index=True
)
test_loader = DataLoader(dataset=CoughDataSet(test_sample), batch_size=TEST_CASES, shuffle=True)

accu_list = []
fpr, tpr, thresholds_roc, precision, recall, thresholds_prc = [], [], [], [], [], []
auc = 0
for step, (x, y) in enumerate(test_loader):
    pred_val = black_box(x.to(DEVICE).view(-1, 1, 64, 16))

    y_true = y.to('cpu').numpy()
    y_scores = pred_val.to('cpu').detach().numpy()[:, 1] - pred_val.to('cpu').detach().numpy()[:, 0]

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores, pos_label=1)
    auc = roc_auc_score(y_true, y_scores)

    precision, recall, thresholds_prc = precision_recall_curve(y_true, y_scores)

# %%

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(
    fpr,
    tpr,
    '-',
    np.linspace(0, 1, 1000),
    np.linspace(0, 1, 1000),
    '--'
)
ax1.set_ylim(0, 1)
ax1.set_xlim(0, 1)
ax1.set_ylabel('True Positive Rate')
ax1.set_xlabel('False Positive Rate')
ax1.legend(['CNN Classifier AUC=%.4f' % auc, 'Random Classifier'])

ax2.plot(
    recall,
    precision,
    '-',
    np.linspace(0, 1, 1000),
    np.repeat(0.5, 1000),
    '--'
)
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylabel('Precision')
ax2.set_xlabel('Recall')
ax2.legend(['CNN Classifier', 'Random Classifier'])

fig.tight_layout()
fig.show()
fig.savefig('./figures/eval-%s-epoch%d.png' % (RUN, MODEL))

with open('./results/%s-%d-roc-threshold.pkl' % (RUN, MODEL), 'wb') as handle:
    pickle.dump(thresholds_roc, handle)

with open('./results/%s-%d-prc-threshold.pkl' % (RUN, MODEL), 'wb') as handle:
    pickle.dump(thresholds_prc, handle)





