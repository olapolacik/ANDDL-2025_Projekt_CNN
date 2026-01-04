import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, activation='relu', conv_layers=2):
        super().__init__()
        # Wybór funkcji aktywacji
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            act_fn = nn.ReLU()

        # Jedna lub dwie warstwy konwolucyjne
        if conv_layers == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                act_fn,
                nn.MaxPool2d(2)
            )
            fc_input = 32 * 13 * 13
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                act_fn,
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                act_fn,
                nn.MaxPool2d(2)
            )
            fc_input = 64 * 5 * 5

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 128),
            act_fn,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
