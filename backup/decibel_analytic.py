import os
import pandas as pd

import pickle
from tqdm import tqdm


BASE_PATH = '/home/jlchen/sandbox/'
TRAIN_LABEL_PATH = './label/segmented_train_set_balanced.csv'
VAL_LABEL_PATH = './label/segmented_val_set_balanced.csv'

label_list = pd.read_csv(TRAIN_LABEL_PATH, index_col=0)

cough_rms = []
non_cough_rms = []

for index, row in tqdm(label_list.iterrows(), total=label_list.shape[0]):
    record_label = False
    if row['label'] > 0.9:
        record_label = True
    elif row['label'] == 0:
        record_label = False
    else:
        continue

    audio_id = row['audio']
    filepath = os.path.join(BASE_PATH, "feature/dnn_paper/dnn2016_%d.pkl" % audio_id)
    try:
        with open(filepath, 'rb') as handler:
            data = pickle.load(handler)
            if record_label:
                cough_rms.append(data['rms'][int(row['window_index'])])
            else:
                non_cough_rms.append(data['rms'][int(row['window_index'])])
    except Exception as e:
        print(e)
        exit(0)
        pass

with open('./results/cough_rms_train.pkl', 'wb') as handler:
    pickle.dump(cough_rms, handler)

with open('./results/non_cough_rms_train.pkl', 'wb') as handler:
    pickle.dump(non_cough_rms, handler)



