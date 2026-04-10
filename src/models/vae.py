import torch
import torch.nn as nn

from src.config import SEQ_LENGTH


class VAE(nn.Module):
    def __init__(self, input_dim=128, seq_length=SEQ_LENGTH, latent_dim=32, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        flat_dim = seq_length * input_dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def encode(self, x):
        h = self.encoder(x.reshape(x.size(0), -1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder(z)
        out = out.view(z.size(0), self.seq_length, self.input_dim)
        return torch.sigmoid(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
