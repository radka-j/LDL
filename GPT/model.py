# =================================================================================================
# MODEL
#
# Throughout use B,T,C to indicate dimensions:
# B = BATCH_SIZE
# T = TIME (BLOCK_SIZE i.e., where in the sequence we are within the allowed sequence limit)
# C = CHANNELS (this was VOCAB_SIZE in bigram model, now it is N_EMND i.e., embedding dimension)
#
# The network is quite deep, need 2 additoinal steps to get around optimisation issues
# - 1: skip/residual connections
# - 2: layer norm
#
# Lasty, also include dropout (there are a few places throughout where this is applied)
# =================================================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Self-attention head
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # this is a triangular matrix, 1s on the lower triangular and 0s in upper
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B,T,C) where T=BLOCK_SIZE and C=N_EMBD
        _, T, _ = x.shape

        # each token emits a key, query and a value vector
        # query: the search i.e., what I am looking for
        # key: what do I contain i.e., context, tags, etc.
        # value: the thing you are searching for
        # all are (B,T,C) where C = head_size
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # compute attention scores ("affinities")
        # for matrix multiplication need to transpose and handle the channel dimension
        # attention also gets scaled by sqrt of head_size to control variance at initialization
        head_size = k.size(-1)
        att = (
            q @ k.transpose(-2, -1) * head_size**-0.5
        )  # (B, T, C) @ (B, C, T) -> (B, T, T)

        # the below line makes this a decoder block (i.e., don't communicate with future tokens)
        # - if take out, have an encoder
        # the matrix has values in the lower triangular and -inf in the upper
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)

        # softmax turns -inf to 0 and normalizes the remaining values
        att = F.softmax(att, dim=-1)  # (B, T, T)

        # dropout i.e., randomly prevent some of the nodes/tokens from communicating with each other
        att = self.dropout(att)

        # perform weighted aggregation of the values (not raw inputs)
        out = att @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHead(nn.Module):
    """
    Multi-head attention: multiple self-attention heads processed in parallel
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        # call this a projection rather than transformation because dims are unchanged
        # (a projection goees back to same vector space and leaves the image unchanged)
        # just following the paper implementation here
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate the individual head outputs over the channels dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FNN(nn.Module):
    """
    Feed-forward neural net: two linear transformations with a ReLU activation in between
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        # define structure
        self.net = nn.Sequential(
            # in the paper they specify that the input/output embedding dimensions are 512
            # while the inner layer has dimensionality 2048
            # so multiply by 4 below to reflect the same factor
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # for ease include dropout here, otherwise could apply as a separate step as above
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: a block has N heads running in parallel followed by a FFN
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # due to having multiple heads, reduce head size to reduce computation
        # if have a single head - this stays n_embd sized (same as in paper)
        head_size = n_embd // n_head
        # self-attention i.e., the "communication part"
        self.sa = MultiHead(n_head, head_size, n_embd, block_size, dropout)
        # feed-forward net i.e., the "computation part"
        self.ffwd = FNN(n_embd, dropout)
        # apply layer norm (at each step within block)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # NOTE: each step also implements a residual connection (due to the + operation)

        # 1. apply self attention
        x = x + self.sa(self.ln1(x))
        # 2. pass through feed-forward NN
        x = x + self.ffwd(self.ln2(x))

        return x


class GPT(nn.Module):
    """
    Decoder-only transformer.
    """

    def __init__(
        self, vocab_size, n_embd, block_size, n_head, n_blocks, dropout, device
    ):
        super().__init__()
        #  token look up table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # need also a position table to track position info
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # create sequence of N_BLOCKS with N_HEAD each
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_blocks)]
        )
        # final layer norm - apply after all blocks
        self.ln_f = nn.LayerNorm(n_embd)
        # last linear transformation to go from n_embd to vocab_size dimension
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, inputs_idx, targets_idx=None):
        """
        - function to evaluate model given inputs and expected output
        - returns logits over tokens and the loss (unless targets not provided)
        - remember that inputs and targets are tensors of indexes of size (B,T)
        - C here is N_EMBD
        """
        B, T = inputs_idx.shape

        # get logits for what is coming next in the sequence
        tok_emb = self.token_embedding_table(inputs_idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)

        # COMBINE TOKEN and POSITION EMBEDDINGS
        x = tok_emb + pos_emb  # (B,T,C)
        # pass through all blocks
        x = self.blocks(x)  # (B,T,C)
        # pas through the final linear layer
        x = self.ln_f(x)  # (B,T,C)
        # lastly, return logits over tokens (rather than embedding dimension)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # when generating, don't have targets
        if targets_idx is None:
            loss = None
        else:
            # how well are we predicting the nex ttoken compared to the targets?
            # the cross_entrypy function expects channels to be the second dimension
            # --> reshape the logits tensor to be (batch*time, channels) to comply with this
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets_idx = targets_idx.view(B * T)
            loss = F.cross_entropy(logits, targets_idx)

        return logits, loss

    def generate(self, context, max_new_tokens, block_size):
        """
        Generate max_new_tokens given inputs.
        """
        # context is a (B,T) tensor

        # here loop over max_new_tokens, at each step take the last block_size items
        # and generate the next item --> append this to context

        for _ in range(max_new_tokens):
            # use only the last block_size tokens
            idx_cond = context[:, -block_size:]
            # generate logits over next tokens
            logits, loss = self(idx_cond)
            # focus only on the last timestep i.e., get (B,C) tensor
            logits = logits[:, -1, :]
            # apply softmax to turn logits into probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution given the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,C)
            # append sampled index to the running sequence
            context = torch.cat((context, idx_next), dim=1)  # (B, T+1)

        return context
