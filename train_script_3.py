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
from components.model import GPTModel
from components.dataset import TextDataset

# Load dataset
dataset = load_dataset("starhopp3r/TinyChat")
print(
    dataset["train"][100]["text"][:500]
)  # Print the first 500 characters of the first article
print(dataset["train"][600000])

tokenizer = tiktoken.get_encoding("gpt2")

base_encoding = tiktoken.get_encoding("gpt2")

special_tokens = {
    "[INST]": base_encoding.n_vocab,  # next available token id
    "[/INST]": base_encoding.n_vocab + 1,
}

# 3. Create a new encoding that merges GPT‑2’s tokens + your special tokens
tokenizer = tiktoken.Encoding(
    name="gpt2_with_inst",
    pat_str=base_encoding._pat_str,
    mergeable_ranks=base_encoding._mergeable_ranks,
    special_tokens={**base_encoding._special_tokens, **special_tokens},
)


def encode(text):
    return tokenizer.encode(text, allowed_special={"[INST]", "[/INST]"})


def decode(tokens):
    return tokenizer.decode(tokens)


print("testing encoding and decoding functions:")
print(encode("[INST] Hello, world! [/INST]"))
print(decode(encode("[INST] Hello, world! [/INST]")))


# hyperparameters
train_model = True
periodic_outputs = False
block_size = 128
n_layers = 16
n_heads = 8
dropout_p = 0.1
batch_size = 64
learning_rate = 3e-4
n_embedding = 256
max_iters = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = TextDataset(dataset, block_size=block_size)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16
)


# define objects
vocab_size = tokenizer.n_vocab

model = GPTModel(vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size).to(
    device
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


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
            # dataloader exhausted — restart it
            data_iter = iter(train_dataloader)
            xb, yb = next(data_iter)

        if count % 100 == 0 and periodic_outputs:
            # print out xb, yb, encoded too
            print("xb decoded: ", decode(xb[0].tolist()))
            print("yb decoded: ", decode(yb[0].tolist()))
            print("---" * 10)
            print("xb raw: ", xb[0].tolist())
            print("yb raw: ", yb[0].tolist())
        #
        # except StopIteration:
        #     break  # dataloader exhausted before max_iters

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
def generate_text(model, prompt, max_new_tokens, block_size, device):
    model.eval()
    # Encode the prompt text into token IDs using our custom encode function
    tokens = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

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

    # Decode back into text using our custom decode function
    output_tokens = tokens[0].tolist()
    output_text = decode(output_tokens)
    return output_text


# print model parameters
print(
    f"Model has {sum(p.numel() for p in model.parameters()) / 1000000:.6f} million parameters."
)
prompt = "this new company does [/INST]"
print(
    generate_text(
        model, prompt, max_new_tokens=500, block_size=block_size, device=device
    )
)
