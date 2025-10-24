import torch
import torch.nn as nn
from torch.nn import functional as F
import math, time, os
from torch.utils.data import Dataset, DataLoader
import tiktoken

# from torch.cuda.amp import autocast, GradScaler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from datasets import load_dataset

# dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
dataset = load_dataset("Bingsu/openwebtext_20p")
# This gives you cleaned, plain text articles1
print(
    dataset["train"][100]["text"][:500]
)  # Print the first 500 characters of the first article
print(dataset["train"][600000])


class TextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.dataset["train"])


    def __getitem__(self, idx):
        # choose a random index instead of using the passed idx
        rand_idx = torch.randint(0, len(self.dataset['train']), (1,)).item()
        tokens = self.tokenizer.encode(self.dataset['train'][rand_idx]['text'])

        if len(tokens) < self.block_size + 1:
            tokens = F.pad(torch.tensor(tokens), (0, self.block_size + 1 - len(tokens)), value=0)
        else:
            tokens = torch.tensor(tokens[: self.block_size + 1])

        x = tokens[: self.block_size]
        y = tokens[1 : self.block_size + 1]
        return x.long(), y.long()

# hyperparameters
train_model = True
block_size = 256
n_layers = 32
n_heads = 16
dropout_p = 0.1
batch_size = 32
learning_rate = 3e-4
n_embedding = 512
max_iters = 50000
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = TextDataset(dataset, tokenizer, block_size=128)
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, drop_last=True
)


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


# define objects
vocab_size = tokenizer.n_vocab

model = GPTModel(vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size).to(
    device
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

from tqdm import tqdm

# training loop
torch.set_float32_matmul_precision("high")
scaler = GradScaler(device)
if train_model:
    compiled_model = torch.compile(model)

    pbar = tqdm(range(max_iters), desc="Training", ncols=100)
    data_iter = iter(train_dataloader)

    for count in pbar:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            break  # dataloader exhausted before max_iters

        xb, yb = xb.to(device), yb.to(device)
        # logits = compiled_model(xb)
        # loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        with autocast(device, dtype=torch.float16):
            logits = compiled_model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

        # backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update bar text dynamically
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

if train_model:
    torch.save(model.state_dict(), "checkpoints/gpt_model-1.pth")
else:
    model.load_state_dict(torch.load("checkpoints/gpt_model-1.pth"))


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens, block_size, device):
    model.eval()
    # Encode the prompt text into token IDs
    tokens = (
        torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    )

    for _ in range(max_new_tokens):
        # Only keep the last block_size tokens for context
        input_tokens = tokens[:, -block_size:]

        # Get logits and take the last token's distribution
        logits = model(input_tokens)
        logits = logits[:, -1, :]  # (batch=1, vocab)
        probs = F.softmax(logits, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)

    # Decode back into text
    output_text = tokenizer.decode(tokens[0].tolist())
    return output_text


prompt = "Once upon a thing was"
print(
    generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=50,
        block_size=block_size,
        device=device,
    )
)

