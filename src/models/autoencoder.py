import torch
import torch.nn as nn

from src.config import SEQ_LENGTH


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).repeat(1, SEQ_LENGTH, 1)
        decoded, _ = self.decoder(repeated)
        return self.output_proj(decoded)
