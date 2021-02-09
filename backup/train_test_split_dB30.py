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

# %%
# Read Current Labels
table_fine = pd.read_csv(os.path.join(BASE_PATH, './fine_grained_annotation.csv'), index_col=0)
id_true_fine_list = table_fine[table_fine['label'] == 'Cough']['coarse_grained_annotation_id'].unique()
id_fine_list = table_fine['coarse_grained_annotation_id'].unique()

table_coarse = pd.read_csv(os.path.join(BASE_PATH, 'coarse_grained_annotation.csv'), index_col=0)
id_true_coarse_list = table_coarse[table_coarse['label'] == True]['id'].unique()
id_coarse_list = table_coarse['id'].unique()
id_false_coarse_list = list(set(id_coarse_list) - set(id_true_coarse_list) - set(id_true_fine_list))

id_all = list(set(id_fine_list).union(set(id_coarse_list)))

print(
    len(id_false_coarse_list),
    len(id_true_fine_list),
    len(id_all)
)

# %%
# Check Valid Files
'''
id_broken = []
missing = 0
audio_error = 0
for audio_id in tqdm(id_all, total=len(id_all)):

    file_path = os.path.join(BASE_PATH, "feature/dnn_paper/dnn2016_%d.pkl" % audio_id)
    try:
        with open(file_path, 'rb') as feature_handler:
            data = pickle.load(feature_handler)
    except OSError as e:
        missing += 1
        id_broken.append(audio_id)
    except Exception as e:
        audio_error += 1
        id_broken.append(audio_id)

# %%
print(
    missing,
    audio_error,
)

print(
    id_broken
)
exit(0)
'''
# %%
id_broken = [387, 389, 402, 1079, 1081]

# %%
id_false_coarse_list_valid = list(set(id_false_coarse_list) - set(id_broken))
id_true_fine_list_valid = list(set(id_true_fine_list) - set(id_broken))

# %%
table_ready = pd.concat(
    [
        pd.DataFrame({
            "audio_id": id_false_coarse_list_valid,
            "label": False
        }),
        pd.DataFrame({
            "audio_id": id_true_fine_list_valid,
            "label": True
        })
    ],
    ignore_index=True
).set_index('audio_id', drop=True)

# %%
labels_temp, labels_test = train_test_split(
    table_ready,
    test_size=0.3,
    random_state=SEED,
    shuffle=True,
)
labels_train, labels_val = train_test_split(
    labels_temp,
    test_size=0.25,
    random_state=SEED,
    shuffle=True,
)


'''
# Plot decibel
                    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,9))
                    db = 20 * (np.log10(data['rms']) - 2.07918124605)
                    ax1.plot(
                        np.arange(0, audio_length),
                        db
                    )
                    ax1.set_xlim(0, audio_length)
                    ax2.imshow(data['zxx_log'], aspect="auto", origin='lower', cmap="inferno")
                    fig.tight_layout()
                    fig.show()
'''


def rms_to_db(rms):
    return 10 * np.square(np.log10(rms))

# %%
def label_segmentation(label_list):
    df_list = []

    for index, row in tqdm(label_list.iterrows(), total=label_list.shape[0]):
        record_label = row['label']
        audio_id = index
        filepath = os.path.join(BASE_PATH, "feature/dnn_paper/dnn2016_%d.pkl" % audio_id)
        try:
            with open(filepath, 'rb') as handler:
                data = pickle.load(handler)
                audio_length = data['zxx_log'].shape[1]
                decibel = rms_to_db(data['rms'])

                # COUGH
                if record_label:
                    # Load Fine Grained Data
                    record_list = table_fine.loc[table_fine.coarse_grained_annotation_id == audio_id, ['label', 'label_end', 'label_start']]
                    cough_list = record_list[record_list.label == 'Cough']

                    # Encode One-hot Array for the cough event
                    time_seq = np.repeat(False, audio_length)
                    for index, cough in cough_list.iterrows():
                        label_start = round(cough['label_start'] * SR) // WINDOW_SIZE
                        label_end = round(cough['label_end'] * SR) // WINDOW_SIZE
                        time_seq[label_start: label_end - 1] = True

                    # Sliding Window
                    padded = np.pad(time_seq, (0, 16), constant_values=(0, 0))
                    rolling_list = np.array(
                        (
                            padded,
                            np.roll(padded, -1),
                            np.roll(padded, -2),
                            np.roll(padded, -3),
                            np.roll(padded, -4),
                            np.roll(padded, -5),
                            np.roll(padded, -6),
                            np.roll(padded, -7),
                            np.roll(padded, -8),
                            np.roll(padded, -9),
                            np.roll(padded, -10),
                            np.roll(padded, -11),
                            np.roll(padded, -12),
                            np.roll(padded, -13),
                            np.roll(padded, -14),
                            np.roll(padded, -15),
                        )
                    )

                    labels = np.sum(rolling_list, axis=0) / 16.0

                    df = pd.DataFrame({
                            "audio": audio_id,
                            "window_index": np.arange(0, labels[:-16 - 15].shape[0]),
                            "label": labels[:-16 - 15],
                            "db": decibel[:- 15]
                    })

                    df = df[df.db >= 30.0].drop(columns=['db'])

                    df_list.append(df)

                    # NON-COUGH
                else:
                    df = pd.DataFrame({
                            "audio": audio_id,
                            "window_index": np.arange(0, audio_length - 15),
                            "label": 0,
                            "db": decibel[:- 15]
                    })

                    df = df[df.db >= 30.0].drop(columns=['db'])

                    df_list.append(df)

                    pass
        except Exception as e:
            pass

    return pd.concat(df_list, ignore_index=True)


# %%
label_segmentation(labels_train).to_csv('./label/segmented_train_set_dB30.csv')
label_segmentation(labels_val).to_csv('./label/segmented_val_set_dB30.csv')

# %%
labels_test.to_csv('./label/test_set.csv')

