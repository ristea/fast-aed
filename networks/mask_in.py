import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
import numpy as np
import matplotlib.pyplot as plt


class MaskIn(nn.Module):
    def __init__(self, patch_dim, percentage_mask=0.3):
        super().__init__()
        self.percentage_mask = percentage_mask
        self.unfold = Rearrange('b c (p1 w) (p2 h) -> b c (p1 p2) w h', p1=patch_dim, p2=patch_dim)
        self.fold = Rearrange('b c (p1 p2) w h -> b c (p1 w) (p2 h)', p1=patch_dim, p2=patch_dim)
        self.indexes = np.array(range(0, patch_dim**2))

    def forward(self, x):
        np.random.shuffle(self.indexes)
        mask_indexes = self.indexes[:int(len(self.indexes) * self.percentage_mask)]
        mask = torch.ones_like(x)

        x_unfold = self.unfold(x)
        mask = self.unfold(mask)
        mask[:, :, mask_indexes] = 0.

        x_masked = x_unfold * mask
        return self.fold(x_masked)


if __name__ == '__main__':
    mask_module = MaskIn(16, 0.3)
    input = 1+torch.randn(3, 3, 128, 128)
    out = mask_module(input)

    plt.figure("input")
    plt.imshow(input.numpy()[0, 0], cmap='gray')
    plt.figure("output")
    plt.imshow(out.numpy()[0, 0], cmap='gray')
    plt.show()
    print(1)