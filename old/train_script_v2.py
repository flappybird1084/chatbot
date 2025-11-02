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
from tqdm import tqdm


# Load dataset
dataset = load_dataset("Bingsu/openwebtext_20p")
# This gives you cleaned, plain text articles1
print(dataset['train'][100]['text'][:500])  # pyright: ignore[reportArgumentType] # Print the first 500 characters of the first article
print(dataset['train'][600000]) # pyright: ignore[reportArgumentType]


class TextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        # Start with a random index sample
        rand_idx = torch.randint(0, len(self.dataset['train']), (1,)).item()
        text = self.dataset['train'][rand_idx]['text']
        tokens = self.tokenizer.encode(text)

        # Keep appending more samples if too short
        while len(tokens) < self.block_size + 1:
            next_idx = torch.randint(0, len(self.dataset['train']), (1,)).item()
            next_text = self.dataset['train'][next_idx]['text']
            tokens.extend(self.tokenizer.encode(" " + next_text))
            # Prevent runaway growth
            if len(tokens) > self.block_size * 2:
                break

        # Truncate to block_size + 1
        tokens = torch.tensor(tokens[: self.block_size + 1])

        x = tokens[: self.block_size]
        y = tokens[1 : self.block_size + 1]
        return x.long(), y.long()


# hyperparameters
train_model = True
block_size = 256
n_layers = 8
n_heads = 8
dropout_p = 0.1
batch_size = 8
learning_rate = 3e-4
n_embedding = 512
max_iters = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding = nn.Embedding(block_size, n_embedding)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embedding, nhead=n_heads, dropout=dropout_p)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embedding)
        self.head = nn.Linear(n_embedding, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.block_size = block_size

    def forward(self, x):
        bsz, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# Initialize tokenizer and dataset
tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = TextDataset(dataset, tokenizer, block_size=block_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

# Define model objects
vocab_size = tokenizer.n_vocab

model = GPTModel(vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


# Training loop
def train():
    torch.set_float32_matmul_precision('high')
    scaler = GradScaler(device)
    if train_model:
        compiled_model = torch.compile(model)

        pbar = tqdm(range(max_iters), desc="Training", ncols=100)
        data_iter = iter(train_dataloader)

        for count in pbar:
            xb, yb = next(data_iter)
            
            xb, yb = xb.to(device), yb.to(device)
            
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


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens, block_size, device):
    model.eval()
    # Encode the prompt text into token IDs
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

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


def save_model(model, filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model


def main():
    if train_model:
        train()
        save_model(model, "checkpoints/gpt_model-1.pth")
    else:
        model.load_state_dict(torch.load("checkpoints/gpt_model-1.pth"))

    # Example of generating text after training or loading
    prompt = "me when the "
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=50, block_size=block_size, device=device)
    print(generated_text)


if __name__ == "__main__":
    main()