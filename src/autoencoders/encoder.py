from torch import nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.act_1 = nn.GELU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.act_2 = nn.GELU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.act_3 = nn.GELU()
        self.pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.act_4 = nn.GELU()
        self.pool_4 = nn.MaxPool2d(kernel_size=2)

        self.dense = nn.Linear(1024, 2)

    def forward(self, X):
        out = self.pool_1(self.act_1(self.conv_1(X)))
        out = self.pool_2(self.act_2(self.conv_2(out)))
        out = self.pool_3(self.act_3(self.conv_3(out)))
        out = self.pool_4(self.act_4(self.conv_4(out)))
        out = self.dense(nn.Flatten()(out))
        return out



