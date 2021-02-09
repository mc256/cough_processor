import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

BATCH_SIZE = 20
INPUT_PIXEL_WIDTH = 8
SEED = 2333333

DEVICE = 'cuda:0'
RUN = 'Dec21__dB30_ratio49'

MODEL = 45


DATA_PATH = '/home/jlchen/sandbox/feature/dnn_paper'

test = pd.read_csv('./results/cross_validation/elastic_%s_epoch_%d.csv' % (RUN, MODEL), index_col=0)

y_scores = test['pred'].to_numpy()
y_scores[y_scores == float('-inf')] = y_scores[y_scores != float('-inf')].min()
precision, recall, thresholds_prc = precision_recall_curve(test['actual'].to_numpy(), test['pred'].to_numpy())

fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))
ax2.plot(
    recall,
    precision,
    '-',
)
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylabel('Precision')
ax2.set_xlabel('Recall')
ax2.legend(['CNN Classifier'])

fig.tight_layout()
fig.savefig('./figures/prc_eval-%s-epoch%d.png' % (RUN, MODEL))
fig.show()
