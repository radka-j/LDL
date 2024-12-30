import torch

from utils import get_batch, estimate_loss
from model import GPT


# ====================================
# DATA
# tokenize at CHAR level i.e., get index value for each CHAR
# define loopup tables to go int->char and char->int
# ====================================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
unique_chars = sorted(list(set(text)))

# encode: char -> int
encode_lookup = {char: i for i, char in enumerate(unique_chars)}
# decode: int -> char
decode_lookup = {i: char for i, char in enumerate(unique_chars)}


def encode(string):
    return [encode_lookup[c] for c in string]


def decode(list_ints):
    return "".join([decode_lookup[i] for i in list_ints])


data = torch.tensor(encode(text), dtype=torch.long)

# train, test split (first 90% will be train, rest val)
split = 0.9
n = int(split * len(data))
train_data = data[:n]
val_data = data[n:]


# ====================================
# CONFIG
# ====================================

# model
VOCAB_SIZE = len(unique_chars)  # number of tokens
BATCH_SIZE = 16  # how many sequenes to process in parallel
BLOCK_SIZE = 32  # the max length of a sequence
N_EMBD = 64  # the embedding dimension, size of token embeddings (instead of VOCAB_SIZE)
N_HEAD = 4  # number of heads (run in parallel) within each block
N_BLOCKS = 4  # number of blocks (called n_layer elsewhere)
DROPOUT = 0.0  # % of neurons to randomly turn off during training each forward pass

# optimisation and training
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 5000  # how many steps to train for
eval_interval = 100  # compute estimated loss each eval_iter steps

# ====================================
# TRAIN
# ====================================

# instantiate model
model = GPT(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, N_HEAD, N_BLOCKS, DROPOUT, device)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val data sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)

    # make predictions and evaluate the loss
    logits, loss = model(xb, yb)

    # zero out gradients otherwise will accumulate over iterations
    optimizer.zero_grad(set_to_none=True)

    # compute gradients and then update parameters
    loss.backward()
    optimizer.step()


# ====================================
# GENERATE
# ====================================

# 0 = new line character so start with input 0 and generate 2000 tokens from there
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generate acts on batches (here we have batch 1) --> index into it
print(
    decode(m.generate(context, max_new_tokens=2000, block_size=BLOCK_SIZE)[0].tolist())
)
