import torch.nn as nn


class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        ### FILL: add more CONV Layers
        # Decoder (Deconvolution Layers)
        ### FILL: add ConvTranspose Layers
        ### Note: since last layer outputs RGB channels, may need specific activation function
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, 7, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.upscore_pool5 = nn.Sequential(
            nn.Conv2d(512, 3, 1),
            nn.ConvTranspose2d(3, 3, 4, 2, padding=1)
        )
        self.score_pool4 = nn.ConvTranspose2d(512, 3, 1)
        self.score_pool3 = nn.ConvTranspose2d(256, 3, 1)
        self.upscore_pool4 = nn.ConvTranspose2d(3, 3, 4, 2, padding=1)
        self.upscore_pool = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 16, 8, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        # Decoder forward pass
        ### FILL: encoder-decoder forward pass
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        pool5 = self.upscore_pool5(x7)
        pool4 = self.score_pool4(x4)
        pool3 = self.score_pool3(x3)
        pool4 = pool4 + pool5
        pool4 = self.upscore_pool4(pool4)
        pool = pool3 + pool4
        output = self.upscore_pool(pool)
        return output
    