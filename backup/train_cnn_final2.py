import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from component.cnn_model import ModelC
from component.dataloader_multiprocessing_eval import CoughDataSet
from component.decision_maker import prepare_eval_data, evaluate_result
from component.label_generator import get_labels
CORE = 24
BATCH_SIZE = CORE * 6
SEED = 2333333
num_epochs = 5

TRAIN_SIZE = 0.25
TEST_SIZE = 0.5

DEVICE = 'cuda:0'
# DEVICE = 'cpu'
RUN = 'Dec25_'

DATA_PATH = '/home/jlchen/sandbox/feature/dnn_paper'

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    writer = SummaryWriter(filename_suffix="cnn_cough_%s" % RUN)

    for i in [3, 4]:
        segmented_label_train, label_val = get_labels(i, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

        round_loss = []

        black_box = ModelC().to(DEVICE).double()

        loader = DataLoader(dataset=CoughDataSet(segmented_label_train), batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=CORE, pin_memory=True)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(black_box.parameters(), lr=0.001, momentum=0.9, nesterov=True, dampening=0)
        black_box.train()

        for epoch in range(num_epochs):
            with tqdm(total=len(loader)) as progress_bar:
                for x, y in loader:
                    optimizer.zero_grad()
                    pred_y = black_box(x.to(DEVICE).view(-1, 1, 64, 16))
                    loss = loss_function(pred_y, y.to(DEVICE))
                    loss.backward()
                    optimizer.step()
                    round_loss.append(loss.item())
                    progress_bar.update(1)

                tqdm.write('Epoch %d, Loss:%.4f' % (epoch, np.mean(round_loss).item()))
                writer.add_scalar('Loss/Epoch', np.mean(round_loss).item(), epoch)

        torch.save(black_box.state_dict(), "./model/cnn_final_%s_run_%d.pkl" % (RUN, i))
        black_box.eval()

        y_true = []
        y_scores = []
        for audio_id, row in tqdm(label_val.iterrows(), total=label_val.shape[0]):
            record_label = row['label']

            spectrogram, decibel = prepare_eval_data(audio_id)
            pred_val = black_box(spectrogram.to(DEVICE).view(-1, 1, 64, 16)).to('cpu').detach().numpy()
            result = evaluate_result(pred_val, decibel, 30.0)
            y_true.append(record_label)
            y_scores.append(result)

        y_scores[y_scores == float('-inf')] = y_scores[y_scores != float('-inf')].min()
        y_scores[y_scores == float('-inf')] = y_scores[y_scores != float('-inf')].min()

        pd.DataFrame({
            'pred': y_scores,
            'actual': y_true
        }).to_csv('./results/final_%s_RUN_%d.csv' % (RUN, i))

        try:
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

            fig.tight_layout()
            fig.savefig('./figures/prc_eval-%s-trials%d.png' % (RUN, i))
            fig.show()
        except Exception as e:
            print(e)
            pass
