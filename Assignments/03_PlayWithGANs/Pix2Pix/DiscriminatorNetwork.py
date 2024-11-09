import torch
import torch.nn as nn


class DiscriminatorNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        ### FILL: add more CONV Layers
        # Decoder (Deconvolution Layers)
        ### FILL: add ConvTranspose Layers
        ### Note: since last layer outputs RGB channels, may need specific activation function
        Ck = '''nn.Sequential(
            nn.Conv2d({in_channel}, {out_channel}, 4, 2, 1),
            nn.BatchNorm2d({out_channel}),
            nn.LeakyReLU(0.2, inplace=True),
        )'''
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = eval(Ck.format(in_channel=64, out_channel=128))
        self.conv3 = eval(Ck.format(in_channel=128, out_channel=256))
        self.conv4 = eval(Ck.format(in_channel=256, out_channel=512))
        self.conv5 = nn.Conv2d(512, 1, 4, 2, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):
        # Encoder forward pass
        # Decoder forward pass
        ### FILL: encoder-decoder forward pass
        x = torch.cat((x, cond), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.sigmoid(self.conv5(x)).mean()
        return output
