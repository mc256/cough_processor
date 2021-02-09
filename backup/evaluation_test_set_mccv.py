import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from component.cnn_model import ModelC
from component.decision_maker import prepare_eval_data, evaluate_result

BATCH_SIZE = 20
INPUT_PIXEL_WIDTH = 8
SEED = 2333333

DEVICE = 'cuda:0'

'''
RUN = 'Dec21__dB30_ratio49'
MODEL = 45
'''

'''
RUN = 'Dec21__dB30_ratio99'
MODEL = 26
'''

'''
RUN = 'Dec19-balanced-30dB'
MODEL = 49
'''

RUN = 'Dec21__dB30_ratio19'
MODEL = 49

for RUN, MODEL in [
    # ('Dec21__dB30_ratio49', 45),
    ('Dec21__dB30_ratio99', 26),
    # ('Dec19-balanced-30dB', 49),
    #('Dec21__dB30_ratio19', 49),
]:
    for cv in [0, 1, 2, 3, 4]:
        DATA_PATH = '/home/jlchen/sandbox/feature/dnn_paper'
        TEST_LABEL = './label/test_set.csv'

        black_box = ModelC().to(DEVICE).double()
        print(black_box)

        test_table = pd.read_csv('./label/test_set.csv', index_col=0)

        # print(test_table.head(10))
        test_table = test_table.sample(frac=0.2, random_state=SEED+cv)
        print(test_table.describe())

        black_box.load_state_dict(torch.load('./model/cnn_attempt_%s_epoch_%d.pkl' % (RUN, MODEL)))
        black_box.eval()

        y_true = []
        y_scores = []
        file_id_list = []

        for audio_id, row in tqdm(test_table.iterrows(), total=test_table.shape[0]):
            record_label = row['label']

            spectrogram, decibel = prepare_eval_data(audio_id)
            pred_val = black_box(spectrogram.to(DEVICE).view(-1, 1, 64, 16)).to('cpu').detach().numpy()
            result = evaluate_result(pred_val, decibel, 35.0)
            y_true.append(record_label)
            y_scores.append(result)
            file_id_list.append('feature/dnn_paper/dnn2016_%d.pkl' % audio_id)

        y_scores = np.array(y_scores)
        y_true = np.array(y_true)
        y_scores[y_scores == float('-inf')] = (y_scores[y_scores != float('-inf')]).min()

        output_actual = np.zeros(y_true.shape[0])
        output_actual[y_true] = 1

        pd.DataFrame({
            'file_id': file_id_list,
            'pred': y_scores,
            'actual': output_actual
        }).to_csv('./results/cross_validation/%d---%s_epoch_%d.csv' % (cv, RUN, MODEL))

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
        ax2.set_title('%s - epoch%s %d' % (RUN, MODEL, cv))

        fig.tight_layout()
        fig.savefig('./figures/mccv-%d-prc_eval-%s-epoch%d.png' % (cv,RUN, MODEL))
        fig.show()
