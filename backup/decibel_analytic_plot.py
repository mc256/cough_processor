import numpy as np
import matplotlib.pyplot as plt

import pickle

BASE_PATH = '/home/jlchen/sandbox/'
TRAIN_LABEL_PATH = './label/segmented_train_set_balanced.csv'
VAL_LABEL_PATH = './label/segmented_val_set_balanced.csv'


def rms_to_db(rms):
    return 10 * np.square(np.log10(rms))


with open('./results/cough_db_train.pkl', 'rb') as handler:
    cough_rms = pickle.load(handler)
    cough_db = rms_to_db(cough_rms)
    cough_db = cough_db[cough_db != float("-inf")]
    #cough_db = cough_db[cough_db > 0]
    #cough_db = cough_db[cough_db < 50]

    with open('./results/non_cough_db_train.pkl', 'rb') as handler:
        non_cough_rms = pickle.load(handler)
        non_cough_db = rms_to_db(non_cough_rms)
        non_cough_db = non_cough_db[non_cough_db != float("-inf")]
        #non_cough_db = non_cough_db[non_cough_db > 0]
        #non_cough_db = non_cough_db[non_cough_db < 50]

        n_bins = 1000
        fig, ax = plt.subplots(1, 1, tight_layout=True, sharex=True)

        # We can set the number of bins with the `bins` kwarg
        ax.hist(cough_db, bins=n_bins, density=True, alpha=0.5)
        ax.hist(non_cough_db, bins=n_bins, density=True, alpha=0.5)

        ax.set_xlim([0,50])
        ax.set_ylabel('Density')
        ax.set_xlabel('dB')
        ax.locator_params(axis='x', tight=True, nbins=10)

        ax.legend(['Cough', 'Non-cough'])
        fig.tight_layout()
        fig.show()
        fig.savefig('./figures/decibel_hist.png')


