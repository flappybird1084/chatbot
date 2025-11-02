

import torch
import torch.nn as nn
from torch.nn import functional as F
import math, time, os
from torch.utils.data import Dataset, DataLoader
import tiktoken

# from torch.cuda.amp import autocast, GradScaler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from datasets import load_dataset


class GPTModel(nn.Module):
    def __init__(
        self, vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size
    ):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding = nn.Embedding(block_size, n_embedding)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=n_embedding, nhead=n_heads, dropout=dropout_p
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embedding)
        self.head = nn.Linear(n_embedding, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.block_size = block_size

    def forward(self, x):
        bsz, seq_len = x.size()
        positions = (
            torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        )
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
