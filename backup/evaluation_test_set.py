import os
import torch
import pandas as pd
import numpy as np
import torch as th
import pickle
from tqdm import tqdm

from component.cnn_model import ModelC
from component.dataloader import CoughDataSet

BATCH_SIZE = 20
INPUT_PIXEL_WIDTH = 8
SEED = 2333333

DEVICE = 'cuda:0'
#DEVICE = 'cpu'
RUN = 'Dec19-balanced-30dB'
RUN = 'Dec19-30dB-ratio9'
MODEL = 29

DATA_PATH = '/home/jlchen/sandbox/feature/dnn_paper'
TEST_LABEL = './label/test_set.csv'

THRESHOLD = 0.09637817518481862
THRESHOLD = -1.057206405178274
THRESHOLD = 0.5

black_box = ModelC().to(DEVICE).double()
print(black_box)


def evaluation_an_audio(audio_id):
    file_path = os.path.join(DATA_PATH, "dnn2016_%d.pkl" % audio_id)
    with open(file_path, 'rb') as file_handler:
        data = pickle.load(file_handler)

        compact_window = np.array(
            (
                data['zxx_log'],
                np.roll(data['zxx_log'], -1, axis=1),
                np.roll(data['zxx_log'], -2, axis=1),
                np.roll(data['zxx_log'], -3, axis=1),
                np.roll(data['zxx_log'], -4, axis=1),
                np.roll(data['zxx_log'], -5, axis=1),
                np.roll(data['zxx_log'], -6, axis=1),
                np.roll(data['zxx_log'], -7, axis=1),
                np.roll(data['zxx_log'], -8, axis=1),
                np.roll(data['zxx_log'], -9, axis=1),
                np.roll(data['zxx_log'], -10, axis=1),
                np.roll(data['zxx_log'], -11, axis=1),
                np.roll(data['zxx_log'], -12, axis=1),
                np.roll(data['zxx_log'], -13, axis=1),
                np.roll(data['zxx_log'], -14, axis=1),
                np.roll(data['zxx_log'], -15, axis=1),
            )
        )
        compact_window = np.swapaxes(np.swapaxes(np.swapaxes(compact_window, 0, 1), 0, 2), 1, 2).reshape((-1, 64, 16))

        x = th.from_numpy(compact_window).double()
        pred_val = black_box(x.to(DEVICE).view(-1, 1, 64, 16)).to('cpu').detach().numpy()

        prediction = (pred_val[:, 1] - pred_val[:, 0] >= THRESHOLD)

        prediction_windowed = np.max(np.array(prediction).reshape((-1,16)),axis=1)
        prediction_windowed_shift = np.array(
            (
                prediction_windowed,
                np.roll(prediction_windowed, -1),
                np.roll(prediction_windowed, -2),
                np.roll(prediction_windowed, -3),
                np.roll(prediction_windowed, -4),
                np.roll(prediction_windowed, -5),
                np.roll(prediction_windowed, -6),
                np.roll(prediction_windowed, -7),
                np.roll(prediction_windowed, -8),
                np.roll(prediction_windowed, -9)
            )
        )

        largest_window = np.max(np.sum(prediction_windowed_shift, axis=0))

        return largest_window


test_table = pd.read_csv('./label/test_set.csv', index_col=0)

print(test_table.head(10))
print(test_table.describe())

black_box.load_state_dict(torch.load('./model/cnn_attempt_%s_epoch_%d.pkl' % (RUN, MODEL)))
black_box.eval()

tp, fp, tn, fn = 0, 0, 0, 0
for audio_id, row in tqdm(test_table.iterrows(), total=test_table.shape[0]):
    record_label = row['label']
    prediction = evaluation_an_audio(audio_id) > 3
    if record_label:
        if prediction == record_label:
            tp += 1
        else:
            fp += 1
    else:
        if prediction == record_label:
            tn +=1
        else:
            fn +=1

print(
    "tp", tp,
    "fp", fp,
    "tn", tn,
    "fn", fn,
)

print('acc:', (tp + tn) / (tp+tn+fp+fn))
print('f1:', (2*tp)/(2*tp + fp + fn))
print('Precision:', tp / (tp+fp))
print('Recall:', tp / (tp+fn))

