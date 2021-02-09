import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from component.cnn_model import ModelC
from component.decision_maker import prepare_eval_data, evaluate_result
from component.configuration import DEVICE


EXPERIMENT_NAME = 'Dec26_onboarding_v1testset_'


black_box = ModelC().to(DEVICE).double()
print(black_box)

#test_table = pd.read_csv('label/Dec25_small_nosliding__val_run0.csv', index_col=0)
#test_table = pd.read_csv('label/Dec25_onboarding__val_run0.csv', index_col=0)
test_table = pd.read_csv('label/test_set.csv', index_col=0)

# print(test_table.head(10))
#test_table = test_table.sample(frac=0.5, random_state=SEED)
#print(test_table.describe())

for RUN in [2,3,4]:
    #black_box.load_state_dict(torch.load('./model/cnn_final_Dec25_small_nosliding__run_0.pkl'))
    black_box.load_state_dict(torch.load('model/cnn_final_Dec25_onboarding__run_0.pkl'))
    black_box.eval()

    y_true = []
    y_scores = []
    y_audio = []

    for audio_id, row in tqdm(test_table.iterrows(), total=test_table.shape[0]):
        record_label = row['label']

        spectrogram, decibel = prepare_eval_data(audio_id)
        pred_val = black_box(spectrogram.to(DEVICE).view(-1, 1, 64, 16)).to('cpu').detach().numpy()
        result = evaluate_result(pred_val, decibel, 35.0)
        y_true.append(record_label)
        y_scores.append(result)
        y_audio.append('features/dnn_paper/dnn2016_%d.pkl' % audio_id)


    y_true = np.array(y_true).astype(int)
    y_scores = np.array(y_scores)

    y_scores[y_scores == float('-inf')] = y_scores[y_scores != float('-inf')].min()
    y_scores[y_scores == float('-inf')] = y_scores[y_scores != float('-inf')].min()

    pd.DataFrame({
        'file_id': y_audio,
        'pred': y_scores,
        'actual': y_true
    }).to_csv('./results/cross_validation/mccv_%s_RUN_%d.csv' % (EXPERIMENT_NAME, RUN))

    try:
        precision, recall, thresholds_prc = precision_recall_curve(y_true, y_scores)

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
        fig.savefig('./figures/prc_eval-%s-trials%d.png' % (EXPERIMENT_NAME, RUN))
        fig.show()
    except Exception as e:
        print(e)
        pass
