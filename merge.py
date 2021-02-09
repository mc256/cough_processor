import sys

import pandas as pd
from tqdm import tqdm as tqdm

BATCH_SIZE = 20
INPUT_PIXEL_WIDTH = 8
SEED = 2333333


if __name__ == '__main__':

    if len(sys.argv) == 4:
        label = sys.argv[1]
        result = sys.argv[2]

        label_val = pd.read_csv(sys.argv[1])
        label_pred = pd.read_csv(sys.argv[2], index_col=0)

        error = 0
        count = 0
        df_list = []
        for audio_id, row in tqdm(label_val.iterrows(), total=label_val.shape[0]):
            record_label = row['label']
            row_pred = label_pred.iloc[count]
            count += 1
            if record_label != row_pred['actual']:
                error += 1
            df_list.append(pd.DataFrame({
                'file_id': 'features/dnn_paper/dnn2016_%d.pkl' % audio_id,
                'pred': row_pred['pred'],
                'actual': 1 if record_label else 0
            }, index=[0]))
        print('number of mismatch:', error)
        pd.concat(df_list, ignore_index=True).to_csv(sys.argv[3])
        print('Output file to %s', sys.argv[3])
    else:
        print('python merge.py  [validation_labels.CSV] [prediction.CSV] [output.CSV]')
        print('Example:')
        print('python merge.py  ./label/final_val_run0 ./results/cross_validation/mccv_final_Dec2x__RUN_0.csv ./results/cross_validation/0.csv')