import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    # Modify dimensions here.
    INPUT_DIM = 784  # Do not modify this
    TRANS2_DIM = INPUT_DIM // 2
    TRANS3_DIM = TRANS2_DIM // 2
    TRANS4_DIM = TRANS3_DIM // 2
    LATENT_DIM = 39
    LR_FACTOR = 0.5
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Linear(in_features=self.INPUT_DIM, out_features=self.TRANS2_DIM)
        self.enc2 = nn.Linear(in_features=self.TRANS2_DIM, out_features=self.TRANS3_DIM)
        self.enc3 = nn.Linear(in_features=self.TRANS3_DIM, out_features=self.TRANS4_DIM)
        self.enc4 = nn.Linear(in_features=self.TRANS4_DIM, out_features=self.LATENT_DIM)

        self.dec1 = nn.Linear(in_features=self.LATENT_DIM, out_features=self.TRANS4_DIM)
        self.dec2 = nn.Linear(in_features=self.TRANS4_DIM, out_features=self.TRANS3_DIM)
        self.dec3 = nn.Linear(in_features=self.TRANS3_DIM, out_features=self.TRANS2_DIM)
        self.dec4 = nn.Linear(in_features=self.TRANS2_DIM, out_features=self.INPUT_DIM)

        self.encodings = [
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
        ]

        self.decodings = [
            self.dec1,
            self.dec2,
            self.dec3,
            self.dec4,
        ]

    def encode(self, x):
        for e in self.encodings:
            x = e(x)
            x = nn.LeakyReLU(self.LR_FACTOR)(x)
        return x

    def decode(self, x):
        for d in self.decodings:
            x = d(x)
            x = nn.LeakyReLU(self.LR_FACTOR)(x)
        return x

    def compute_compression_ratio(self, x) -> float:
        input_size = x.numel()
        latent_rep = self.encode(x).numel()
        return input_size / latent_rep

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon

