import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

CSV = 'mccv_Dec30_onboarding_loocv__RUN'
df_list = []
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    df_list.append(
        pd.read_csv('./results/cross_validation/%s_%d.csv' % (CSV, i), index_col=0)
    )

test = pd.concat(df_list, ignore_index=True)

y_scores = test['pred'].to_numpy()
y_scores[y_scores == float('-inf')] = y_scores[y_scores != float('-inf')].min()
precision, recall, thresholds_prc = precision_recall_curve(test['actual'].to_numpy(), test['pred'].to_numpy())

fpr, tpr, thresholds_roc = roc_curve(test['actual'].to_numpy(), test['pred'].to_numpy(), pos_label=1)
auc = roc_auc_score(test['actual'].to_numpy(), test['pred'].to_numpy())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

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
ax1.set_title('Receiver Operating Characteristic')

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
ax2.set_title('Precision-Recall Curve')
fig.tight_layout()
fig.savefig('./figures/auc_prc_%s.png' % (CSV))
fig.show()
