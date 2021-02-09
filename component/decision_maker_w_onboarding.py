import os
import pickle

import numpy as np
import torch as th

from component.configuration import ONBOARDING_DATA_PATH
from component.util import rms_to_db


def prepare_eval_data(audio_id):
    file_path = os.path.join(ONBOARDING_DATA_PATH, "dnn2016_%d.pkl" % audio_id)
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

        return th.from_numpy(compact_window).double(), rms_to_db(data['rms'])
        # pred_val = black_box(x.to(DEVICE).view(-1, 1, 64, 16)).to('cpu').detach().numpy()


def evaluate_result(pred_val, decibel, decibel_lower_bound=0.0):
    prediction = pred_val[:, 1] - pred_val[:, 0]
    prediction[decibel < decibel_lower_bound] = float("-inf")
    prediction[-15:] = float("-inf")

    prediction_windowed = np.max(np.array(prediction).reshape((-1, 16)), axis=1)
    prediction_windowed_shift = np.array(
        (
            prediction_windowed,
            np.roll(prediction_windowed, -1),
            np.roll(prediction_windowed, -2),
            np.roll(prediction_windowed, -3),
            # np.roll(prediction_windowed, -4),
            # np.roll(prediction_windowed, -5),
            # np.roll(prediction_windowed, -6),
            # np.roll(prediction_windowed, -7),
            # np.roll(prediction_windowed, -8),
            # np.roll(prediction_windowed, -9)
        )
    )

    largest_window = np.max(np.min(prediction_windowed_shift, axis=0))

    return largest_window
