from torch import nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense_1 = nn.Linear(in_features=2, out_features=2048)
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        # nn.ConvTranspose2d(128, 64, 4, 2, 0, bias=False)
        self.up_conv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_conv_3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.up_conv_4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)

    def forward(self, X):
        out = self.dense_1(X)
        out = out.view(-1, 128, 4, 4)
        out = self.up_conv_1(out)
        out = self.up_conv_2(out)
        out = self.up_conv_3(out)
        out = self.up_conv_4(out)
        out = self.conv_1(out)
        return out