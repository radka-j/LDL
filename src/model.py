# ====================================
# MODEL
#
# Torch instructions say explicitly to implement the forward method for each module.
# This never gets called, instead just call the module itself.
#
# Throughout use B,T,C to indicate dimensions:
# B = BATCH_SIZE
# T = TIME (BLOCK_SIZE i.e., where in the sequence we are within the allowed sequence limit)
# C = CHANNELS (this was VOCAB_SIZE in bigram model, here N_EMND)
#
# The network is quite deep, need 2 additoinal steps to get around optimisation issues
# - 1: skip/residual connections
#   - have a residual pathway from which can fork off and do some computation and then return to it
#
#       Xt
#       | \
#       |  \
#       |   \
#       |   / <COMPUTATION HAPPES ON THIS PATH>
#       |  /
#       |/
#     <addition>
#       Xt+1
#
#   - addition distributes gradients equally to both branches
#     - so the gradients go through the addition operation directly "from supervision" all the way back to the input
#     - but they also fork off and go through the other branch with the additional computations
#   - the residual blocks (i.e., the computation fork off) are intitialised to contribute little at the beginning
#     so the gradient is unimpeded and just flows
#   - over time, during optimisation, they start to contribute
#   - this helps dramatically with optimisation
#
# - 2: layer norm:
#   - very similar to batch norm
#       - across the batch dimension, make sure any individual neuron has unit Gaussian distribution
#       - i.e., it normalises every column in the bach dimension to have 0 mean and 1 SD
#   - here normalize the rows rather than the columns (TODO: what are the rows here?!)
#   - NOTE: original transform paper applied layer norm after the transform
#       but now it's common to do it before ("pre-norm formulation")
#
# Lasty, also include dropout.
#   - there are a few places throughout where this is applied
#   - used to prevent overfitting, regularisation technique
#   - every forward pass, randomly shut off a subset of neurons (turn them to 0)
#   - sort of trains an emsenble of subnetworks
#   - at inference it is not applied
#
# TODO: go over dimensionality of each layer
# ====================================

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
        # to save parameters that should not be optimised, use "self.register_buffer"
        # this is a triangular matrix, ones on the lower triangular and 0s in upper
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # TODO: C here is head_size ?! which is not embedding dimension...
        B, T, C = x.shape

        # each token emits a key, query and a value vector
        # query: what am I looking for?
        # key: what do I contain?
        # value: TODO
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        v = self.value(x)  # (B,T,C)

        # compute attention scores ("affinities"), all keys communicate with all queries
        # for matrix multiplication need to transpose and handle the channel dimension
        # attention also gets scaled by sqrt of head_size:
        # - the scaling is there to control variance at initialization.
        # - without scaling we might get very divergent values which, when passed through softmax, leads to a very peaky output
        # - i.e, lot of weight given to the large values and very little to the other ones --> avoid this
        att = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)

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
    Multi-head attention:
    - multiple self-attention heads processed in parallel
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        # ModuleList can be indexed like a Python list but it also "does the right torch stuff"
        # presumably that means the parameters within are registered for optimisation etc.
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        # TODO: return linear projection - go from (n_heads, n_embd) to (n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate the individual head outputs over the channels dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # apply linear transformation to the outcome of above layer
        # "this is a projection back into the residual pathway"
        # also apply dropout
        out = self.dropout(self.proj(out))
        return out


class FFN(nn.Module):
    """
    Feed-forward net:
    - a simple linear layer followed by a non-linearity
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
            # below is a projection layer back into the residual pathway
            nn.Linear(4 * n_embd, n_embd),
            # dropout
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: 'communication followed by computation'
    - a block has N heads running in parallel followed by a FFN
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # TODO: something about head_size being chosen for "things to work out"
        head_size = n_embd // n_head
        # self-attention i.e., the "communication part"
        self.sa = MultiHead(n_head, head_size, n_embd, block_size, dropout)
        # feed-forward net i.e.,  the "computation part"
        self.ffwd = FFN(n_embd, dropout)
        # apply layer norm (at each step within block
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # this implements residual connections
        # i.e., have both x and the fork off with computations from which we come back
        # do this at both steps within the block (+ apply layer norm)

        # 1. apply self attention
        x = x + self.sa(self.ln1(x))
        # 2. pass through feed-forward NN
        x = x + self.ffwd(self.ln2(x))

        return x


class GPT(nn.Module):
    """
    Decoder-only transformer.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_blocks, dropout, device):
        # always have to start with nn.Module init
        super().__init__()
        # nn.Embedding is a simple look up table populated with values drawn
        # from standard normal (can access as embedding.weight)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # need also a position table to track position info as this gets lots in attention
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # create N_BLOCKS with N_HEAD each - this is just a sequential operation
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_blocks)])
        # final layer norm - apply after all blocks
        self.ln_f = nn.LayerNorm(n_embd)
        # linear transformation to get output of vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, inputs_idx, targets_idx=None):
        """
        function to evaluate model given inputs and expected output
        returns logits over tokens and the loss (unless targets not provided)
        remember that inputs and targets are tensors of indexes of size (B,T)
        """
        B, T = inputs_idx.shape

        # get logits for what is coming next in the sequence
        # the embedding table returns shape (B,T,C) where C=N_EMBD
        tok_emb = self.token_embedding_table(inputs_idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)

        # COMBINE TOKEN and POSITION EMBEDDINGS
        x = tok_emb + pos_emb  # (B,T,C)
        # pass through all blocks
        x = self.blocks(x)  # (B,T,C)
        # pas through the final linear layer
        x = self.ln_f(x)  # (B,T,C)
        # lastly, return logits over tokens (rather than embedding dimension)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # sometimes want to retun just the logits and don't have targets to compare against
        # happens when we are generating
        if targets_idx is None:
            loss = None
        else:
            # how well are we predicting the next character compared to the targets?
            # the cross_entrypy function expects channels to be the second dimension
            # here we reshape the logits tensor to be (batch*time, channels) to comply with this
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets_idx = targets_idx.view(B * T)
            loss = F.cross_entropy(logits, targets_idx)

        return logits, loss

    def generate(self, context, max_new_tokens, block_size):
        """
        Generate max_new_tokens given inputs.
        """
        # context is (B,T) tensor

        # here loop over max_new_tokens, at each step take the last block_size items
        # and generate the next item --> append this to context

        for _ in range(max_new_tokens):
            # use only the last block_size tokens
            idx_cond = context[:, -block_size:]
            # generate logits over next tokens
            logits, loss = self(idx_cond)
            # focus only on the last timestep i.e., get (B, C) tensor
            logits = logits[:, -1, :]
            # apply softmax to turn logits into probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution given the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,C)
            # append sampled index to the running sequence
            context = torch.cat((context, idx_next), dim=1)  # (B, T+1)

        return context
