import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm

from component.configuration import BASE_PATH, DATA_PATH, SR, SEED, WINDOW_SIZE
from component.util import rms_to_db


def read_coarse_labels():
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
    return table_ready


def label_segmentation(label_list, decibel_lower_bound=0.0):
    df_list = []
    table_fine = pd.read_csv(os.path.join(BASE_PATH, './fine_grained_annotation.csv'), index_col=0)

    for index, row in tqdm(label_list.iterrows(), total=label_list.shape[0]):
        record_label = row['label']
        audio_id = index
        filepath = os.path.join(DATA_PATH, "dnn2016_%d.pkl" % audio_id)
        try:
            with open(filepath, 'rb') as handler:
                data = pickle.load(handler)
                audio_length = data['zxx_log'].shape[1]
                decibel = rms_to_db(data['rms'])

                # COUGH
                if record_label:
                    # Load Fine Grained Data
                    record_list = table_fine.loc[
                        table_fine.coarse_grained_annotation_id == audio_id, ['label', 'label_end', 'label_start']]
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

                    df = df[df.db >= decibel_lower_bound].drop(columns=['db'])

                    df_list.append(df)

                    # NON-COUGH
                else:
                    df = pd.DataFrame({
                        "audio": audio_id,
                        "window_index": np.arange(0, audio_length - 15),
                        "label": 0,
                        "db": decibel[:- 15]
                    })

                    df = df[df.db >= decibel_lower_bound].drop(columns=['db'])

                    df_list.append(df)

                    pass
        except Exception as e:
            pass

    return pd.concat(df_list, ignore_index=True)


def get_labels(exp, run=0, train_size=0.1, test_size=0.1):
    label_train, label_val = train_test_split(
        read_coarse_labels(),
        train_size=train_size,
        test_size=test_size,
        random_state=SEED + run,
        shuffle=True
    )

    segmented_label_train = label_segmentation(label_train, 30.0)

    print(segmented_label_train.describe())

    #segmented_label_train.to_csv('./label/%s_train_run%d.csv' % (exp, run))
    label_val.to_csv('./label/%s_val_run%d.csv' % (exp, run))

    return segmented_label_train, label_val
