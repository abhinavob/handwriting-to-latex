import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, tokens = zip(*batch)

    # stack images
    images = torch.stack(images)

    # convert tokens to tensors
    tokens = [torch.tensor(t) for t in tokens]

    # pad sequences
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)

    return images, tokens