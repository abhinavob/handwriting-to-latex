from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from tokenizer import LatexTokenizer, PAD_ID, SOS_ID, EOS_ID

class ResizeAndPad:
    def __init__(self, target_h=64, max_w=512, pad_value=255):
        self.target_h = target_h
        self.max_w = max_w
        self.pad_value = pad_value

    def __call__(self, img):
        w, h = img.size
        new_w = max(1, int(w * self.target_h / h))
        new_w = min(new_w, self.max_w)

        img = img.resize((new_w, self.target_h), Image.BILINEAR)

        padded = Image.new("RGB", (self.max_w, self.target_h),
                           (self.pad_value, self.pad_value, self.pad_value))
        padded.paste(img, (0, 0))
        return padded

class HandwritingAugment:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

     
        if random.random() < 0.4:
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle, fill=255)

        if random.random() < 0.3:
            img = TF.affine(
                img, angle=0,
                translate=(random.randint(-3, 3), random.randint(-2, 2)),
                scale=random.uniform(0.95, 1.05),
                shear=random.uniform(-3, 3),
                fill=255,
            )

        return img

class HMEDataset(Dataset):
    def __init__(self, root_dir, label_file, tokenizer, max_samples=None,
                 max_len=256, img_h=64, img_w=512, augment=False):                                     
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.samples = []

        with open(label_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break

               
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                img_path, label = parts

                
                token_len = len(tokenizer.tokenize(label))
                if token_len > max_len - 2: 
                    continue

                self.samples.append((img_path, label))

    
        self.resize_pad = ResizeAndPad(target_h=img_h, max_w=img_w)

    
        self.augment_fn = HandwritingAugment(p=0.5) if augment else None

  
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]

        img_path = self.root_dir / img_rel_path
        image = Image.open(img_path).convert('RGB')

        image = self.resize_pad(image)

        if self.augment_fn is not None:
            image = self.augment_fn(image)

        image = self.to_tensor(image) 

       
        token_ids = self.tokenizer.encode(label)

        return image, torch.tensor(token_ids, dtype=torch.long)
        
def collate_fn(batch):
    images, token_seqs = zip(*batch)

    images = torch.stack(images, dim=0)

    max_seq_len = max(seq.size(0) for seq in token_seqs)
    padded = torch.full((len(token_seqs), max_seq_len), PAD_ID, dtype=torch.long)

    for i, seq in enumerate(token_seqs):
        padded[i, :seq.size(0)] = seq

    return images, padded
    
def build_dataloaders(root_dir, train_label_file, val_label_file=None,
                      tokenizer=None, max_samples=None, val_split=0.1,
                      batch_size=32, num_workers=4, img_h=64, img_w=512,
                      max_len=256):
    if tokenizer is None:
        tokenizer = LatexTokenizer()
        labels = []
        with open(train_label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    labels.append(parts[1])
        tokenizer.build_vocab(labels)
        print(f"Built tokenizer — vocab size: {tokenizer.vocab_size}")

    train_ds = HMEDataset(
        root_dir, train_label_file, tokenizer,
        max_samples=max_samples, max_len=max_len,
        img_h=img_h, img_w=img_w, augment=True,
    )

    if val_label_file is not None:
        val_ds = HMEDataset(
            root_dir, val_label_file, tokenizer,
            max_samples=max_samples, max_len=max_len,
            img_h=img_h, img_w=img_w, augment=False,
        )
    else:
        val_size = int(len(train_ds) * val_split)
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    return train_loader, val_loader, tokenizer
       
