# %%
import pandas as pd

# %%
SEED = 2333333
WINDOW_SIZE = 64
SR = 16000

# %%
# Base Path
BASE_PATH = '/home/jlchen/sandbox/'
TRAIN_LABEL_PATH = './label/segmented_train_set_dB30.csv'
VAL_LABEL_PATH = './label/segmented_val_set_dB30.csv'


# %%
def balance_data_set(label_path, ratio=1, positive_threshold=0.9):
    table_labels = pd.read_csv(label_path, index_col=0)
    table_labels.loc[table_labels['label'] >= positive_threshold, ['label']] = 1
    positive_count = table_labels[table_labels['label'] == 1].shape[0]

    positive_samples = table_labels[table_labels['label'] == 1].sample(n=positive_count, random_state=SEED)
    negative_samples = table_labels[table_labels['label'] == 0].sample(n=int(positive_count * ratio), random_state=SEED)

    all_samples = pd.concat((positive_samples, negative_samples), ignore_index=True)
    print(all_samples.describe())

    return all_samples


RATIO = 49
balance_data_set(TRAIN_LABEL_PATH, RATIO).to_csv('./label/segmented_train_set_dB30_ratio%d.csv' % (RATIO))
balance_data_set(VAL_LABEL_PATH, RATIO).to_csv('./label/segmented_val_set_dB30_ratio%d.csv' % (RATIO))
