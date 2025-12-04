"""Encoder model architectures."""
import torch
import torch.nn as nn


class TinyAutoencoder(nn.Module):

    def __init__(self, H, W, in_c=1, out_sz=20):
        super().__init__()

        F1 = 32
        F2 = 32
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(1, F1, kernel_size=3, stride=2,
                            padding=1),  # output BxF1xH/2xW/2
            torch.nn.BatchNorm2d(F1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(F1, F2, kernel_size=3, stride=2,
                            padding=1),  # output BxF2xH/4xW/4
            torch.nn.BatchNorm2d(F2),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(F2 * H // 4 * W // 4, out_sz),
            torch.nn.BatchNorm1d(out_sz),
            torch.nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_sz, F2 * H // 4 * W // 4),  # out: B x (F2*16*16)
            nn.BatchNorm1d(F2 * H // 4 * W // 4),
            nn.LeakyReLU(),
            nn.Unflatten(1, (F2, H // 4, W // 4)),  # out: B x F2 x 16 x 16
            nn.ConvTranspose2d(F2,
                               F1,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # 32x32
            nn.BatchNorm2d(F1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(F1,
                               1,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # 64x64
            nn.Sigmoid(),  # assuming input is normalized to [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

