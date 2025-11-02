import gradio as gr
import torch
import torch.nn.functional as F
from components.model import GPTModel
from components.tokenizer import encode, decode, tokenizer


# -----------------------------
# Load model & configuration
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters should match training
block_size = 128
n_layers = 16
n_heads = 8
dropout_p = 0.1
n_embedding = 256

# initialize model and load weights
vocab_size = tokenizer.n_vocab
model = GPTModel(vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size).to(
    device
)
model.load_state_dict(torch.load("checkpoints/gpt_model-1.pth", map_location=device))
model.eval()


# -----------------------------
# Generation function
# -----------------------------
@torch.no_grad()
def generate_text(prompt, max_new_tokens=200, temperature=1.0, top_k=50):
    model.eval()

    # Wrap message in [INST] and [/INST]
    wrapped_prompt = f"[INST] {prompt.strip()} [/INST]"
    tokens = (
        torch.tensor(encode(wrapped_prompt), dtype=torch.long).unsqueeze(0).to(device)
    )

    inst_token_id = encode("[INST]")[0]

    for _ in range(max_new_tokens):
        input_tokens = tokens[:, -block_size:]
        logits = model(input_tokens)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Stop generation if [INST] appears again (do not include it)
        if next_token.item() == inst_token_id:
            break

        tokens = torch.cat((tokens, next_token), dim=1)

    return decode(tokens[0].tolist())[len(wrapped_prompt) :]


# -----------------------------
# Gradio UI
# -----------------------------
def chat(prompt, max_tokens, temperature, top_k):
    response = generate_text(prompt, max_tokens, temperature, top_k)
    return response


with gr.Blocks(title="TinyChat GPT Model") as demo:
    gr.Markdown("## cute lil chatbot")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt", placeholder="Type your message here...", lines=4
            )
            max_tokens = gr.Slider(10, 500, value=200, step=10, label="Max New Tokens")
            temperature = gr.Slider(0.2, 1.5, value=1.0, step=0.1, label="Temperature")
            top_k = gr.Slider(10, 200, value=50, step=10, label="Top‑K Sampling")
            submit = gr.Button("Generate")

        with gr.Column(scale=3):
            output = gr.Textbox(label="Generated Response", lines=15)

    submit.click(chat, inputs=[prompt, max_tokens, temperature, top_k], outputs=output)

# -----------------------------
# Launch app
# -----------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
