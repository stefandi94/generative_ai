from torch import nn

from src.autoencoders.encoder import Encoder
from src.autoencoders.decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()#.to(device)
        self.decoder = Decoder()#.to(device)

    def forward(self, X):
        out = self.encoder(X)
        out = self.decoder(out)
        return out

