import os

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=11, padding=5),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=9, padding=4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        # self.avg_pool = nn.AdaptiveAvgPool2d((25, 25))

        self.fc = nn.Sequential(
            nn.Linear(8*25*25, 5000),
            nn.LeakyReLU(),
            nn.Linear(5000, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 100*2)
        )

    def forward(self, x):
        x = self.conv(x)
        # x = self.avg_pool(x)
        # print(x.shape)
        # os.system('pause')
        temp = x.shape[0]
        x = torch.flatten(x, start_dim=1)
        # print(x.shape, vector.shape)
        # os.system('pause')
        x = self.fc(x)

        return x.view(temp, 100, 2)