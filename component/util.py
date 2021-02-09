import numpy as np


def rms_to_db(rms):
    return 10 * np.square(np.log10(rms))
