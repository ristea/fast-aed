import torch
import torch.nn as nn


class StageDiscriminator(nn.Module):
    def __init__(self, no_teachers, classes=1, hidden=50):
        super(StageDiscriminator, self).__init__()

        self.large_block = nn.Sequential(
            nn.Conv2d(in_channels=no_teachers, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.medium_block = nn.Sequential(
            nn.Conv2d(in_channels=no_teachers + 4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten()
        )

        self.small_block = nn.Sequential(
            nn.Linear(in_features=no_teachers + 8, out_features=hidden),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden, out_features=classes),
            nn.Sigmoid()
        )

    def forward(self, xl, xm, xs):
        xl = self.large_block(xl)
        xm = torch.cat((xl, xm), 1)

        xm = self.medium_block(xm)
        xs = torch.cat((xm, xs), 1)

        return self.small_block(xs)
