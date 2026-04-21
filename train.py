import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import HMEDataset
from tokenizer import Tokenizer
from dataloader import collate_fn
from encoder import Encoder

from decoder import TransformerDecoder, build_decoder


# ------------------------------------------------------------
# TRAIN ONE EPOCH
# ------------------------------------------------------------

def train_one_epoch(
    model,
    encoder,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip=1.0,
    pad_id=0,
    epoch=1
):

    encoder.eval()
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (images, token_ids) in enumerate(dataloader):

        images = images.to(device)
        token_ids = token_ids.to(device)

        # ---- RUN ENCODER HERE ----
        with torch.no_grad():
            encoder_output, source_lengths = encoder(images)

        # ---- TEACHER FORCING ----
        tgt_input = token_ids[:, :-1]
        tgt_target = token_ids[:, 1:]

        logits = model(tgt_input, encoder_output, source_lengths)

        logits_flat = logits.reshape(-1, model.vocab_size)
        target_flat = tgt_target.reshape(-1)

        loss = criterion(logits_flat, target_flat)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        non_pad = (tgt_target != pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / total_tokens if total_tokens > 0 else 0
            print(f"batch {batch_idx+1} | loss: {avg:.4f}")

    return total_loss / total_tokens if total_tokens > 0 else 0

# Greedy Decode

def greedy_decode(model, encoder, image, tokenizer, device, max_len=60):
    model.eval()
    encoder.eval()

    # --- move image to device ---
    image = image.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # --- run encoder ---
    with torch.no_grad():
        encoder_output, source_lengths = encoder(image)

    # --- start token ---
    start_id = tokenizer.token_to_id["<start>"]
    end_id = tokenizer.token_to_id["<end>"]

    # current sequence (start with <start>)
    generated = torch.tensor([[start_id]], device=device)  # [1, 1]

    for _ in range(max_len):

        # --- forward pass ---
        with torch.no_grad():
            logits = model(generated, encoder_output, source_lengths)

        # take last token prediction
        next_token_logits = logits[:, -1, :]   # [1, vocab_size]

        # greedy choice
        next_token = next_token_logits.argmax(dim=-1)  # [1]

        # append token
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # stop if <end>
        if next_token.item() == end_id:
            break

    # --- convert to tokens ---
    token_ids = generated.squeeze(0).tolist()

    # remove <start>
    token_ids = token_ids[1:]

    # stop at <end>
    if end_id in token_ids:
        token_ids = token_ids[:token_ids.index(end_id)]

    # decode to string
    tokens = [tokenizer.id_to_token[i] for i in token_ids]
    # print(next_token.item())

    return " ".join(tokens)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = args.device

    # ---- DATA ----
    DATA_SIZE = 5000
    DATA_ROOT = "../hme100k"
    TRAIN_FILE = "../hme100k/train.txt"

    # ---- PATHS ----
    checkpoint_path = f"/content/drive/MyDrive/HANDWRITING TO LATEX/files/checkpoint_{DATA_SIZE}.pth"
    model_path = f"/content/drive/MyDrive/HANDWRITING TO LATEX/files/model_{DATA_SIZE}.pth"

    tokenizer = Tokenizer()
    tokenizer.load_vocab("vocab.json")

    dataset = HMEDataset(DATA_ROOT, TRAIN_FILE, tokenizer, DATA_SIZE)

    # ---- SPLIT ----
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ---- LOADERS ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,   # important for decoding
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---- MODELS ----
    encoder = Encoder().to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    decoder, vocab = build_decoder(
        vocab_path="vocab.json",
        device=device
    )

    pad_id = vocab.get("<pad>", 0)

    # ---- LOSS + OPT ----
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- LOAD CHECKPOINT ----
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # ---- TRAIN ----
    for epoch in range(start_epoch, args.epochs + 1):

        t0 = time.time()

        loss = train_one_epoch(
            decoder,
            encoder,
            train_loader,
            optimizer,
            criterion,
            device,
            pad_id=pad_id,
            epoch=epoch
        )

        if scheduler is not None:
            scheduler.step()

        print(f"\nEpoch {epoch} | Loss: {loss:.4f} | Time: {time.time() - t0:.1f}s")
        
        if epoch % 5 == 0:
            # For testing output every 5 epochs
            image, label = dataset[0]
            pred = greedy_decode(decoder, encoder, image, tokenizer, device)
            gt_tokens = [tokenizer.id_to_token[i] for i in label]
            print("GT   :", " ".join(gt_tokens))
            print("PRED :", pred)

        # ---- SAVE CHECKPOINT ----
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, checkpoint_path)

            print(f"Checkpoint saved at epoch {epoch}")

    print("Training complete")

    # ---- SAVE FINAL MODEL ----
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "vocab": vocab
    }, model_path)

    print("Final model saved")

    # ---- TESTING ----
    print("\n--- TESTING ON UNSEEN DATA ---")

    for i, (images, tokens) in enumerate(test_loader):

        image = images[0]
        gt_ids = tokens[0].tolist()

        pred = greedy_decode(decoder, encoder, image, tokenizer, device)

        gt_tokens = [
            tokenizer.id_to_token[t]
            for t in gt_ids
            if t != tokenizer.token_to_id["<pad>"]
        ]

        print(f"\nSample {i}")
        print("GT   :", " ".join(gt_tokens))
        print("PRED :", pred)

        if i == 99:
            break

if __name__ == "__main__":
    main()