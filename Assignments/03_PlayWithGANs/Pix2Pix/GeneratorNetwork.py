import torch
import torch.nn as nn


class GeneratorNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder (Convolutional Layers)
        ### FILL: add more CONV Layers
        Ck_encoder = '''nn.Sequential(
            nn.Conv2d({in_channel}, {out_channel}, 4, 2, 1),
            nn.BatchNorm2d({out_channel}),
            nn.LeakyReLU(0.2, inplace=True),
        )'''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = eval(Ck_encoder.format(in_channel=64, out_channel=128))
        self.conv3 = eval(Ck_encoder.format(in_channel=128, out_channel=256))
        self.conv4 = eval(Ck_encoder.format(in_channel=256, out_channel=512))
        self.conv5 = eval(Ck_encoder.format(in_channel=512, out_channel=512))
        self.conv6 = eval(Ck_encoder.format(in_channel=512, out_channel=512))
        self.conv7 = eval(Ck_encoder.format(in_channel=512, out_channel=512))
        self.conv8 = eval(Ck_encoder.format(in_channel=512, out_channel=512))

        # Decoder (Deconvolution Layers)
        ### FILL: add ConvTranspose Layers
        ### Note: since last layer outputs RGB channels, may need specific activation function
        CDk_decoder = '''nn.Sequential(
            nn.BatchNorm2d({in_channel}),
            nn.Dropout2d(0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d({in_channel}, {out_channel}, 4, 2, 1),
        )'''
        Ck_decoder = '''nn.Sequential(
            nn.BatchNorm2d({in_channel}),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d({in_channel}, {out_channel}, 4, 2, 1),
        )'''
        self.de_conv7 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.de_conv6 = eval(CDk_decoder.format(in_channel=1024, out_channel=1024))
        self.de_conv5 = eval(CDk_decoder.format(in_channel=1536, out_channel=1024))
        self.de_conv4 = eval(CDk_decoder.format(in_channel=1536, out_channel=1024))
        self.de_conv3 = eval(Ck_decoder.format(in_channel=1536, out_channel=512))
        self.de_conv2 = eval(Ck_decoder.format(in_channel=768, out_channel=256))
        self.de_conv1 = eval(Ck_decoder.format(in_channel=384, out_channel=128))
        self.de_conv = eval(Ck_decoder.format(in_channel=192, out_channel=3))
        self.output = nn.Tanh()

    def forward(self, x):
        # Encoder forward pass
        # Decoder forward pass
        ### FILL: encoder-decoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x7 = torch.cat((self.de_conv7(x8), x7), 1)
        x6 = torch.cat((self.de_conv6(x7), x6), 1)
        x5 = torch.cat((self.de_conv5(x6), x5), 1)
        x4 = torch.cat((self.de_conv4(x5), x4), 1)
        x3 = torch.cat((self.de_conv3(x4), x3), 1)
        x2 = torch.cat((self.de_conv2(x3), x2), 1)
        x1 = torch.cat((self.de_conv1(x2), x1), 1)
        output = self.output(self.de_conv(x1))
        return output
