import torch


def get_batch(data, block_size, batch_size):
    """
    Randomly select batch_size number of sequences of length block_size.

    Outputs shape of both inputs and targets is (BATCH_SIZE, BLOCK_SIZE).
    """
    # pick batch_size number of random integers and use those as start index of each batch
    # because of this, choose from integers 0 to len(data)-block_size
    max_idx = len(data) - block_size
    idx = torch.randint(max_idx, (batch_size,))

    # stack the input sequences in the batch
    inputs = torch.stack([data[i : i + block_size] for i in idx])

    # the targets are offset by 1
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])

    return inputs, targets


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=200):
    """ "
    Return loss averaged over eval_iters batches
        - do this for both train and validation data
    NOTE: `torch.no_grad()` disables gradient calculation.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = train_data if split == "train" else val_data
            X, Y = get_batch(data, block_size, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
