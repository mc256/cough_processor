import torch
import torch.nn as nn


class ModelC(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(9, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(1, 1, kernel_size=(5, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        cnn_out = self.cnn_layers(x)
        flatten = torch.flatten(cnn_out, 1)
        classifier = self.fc_layers(flatten)
        return classifier
