from dataset import HMEDataset, collate_fn
from tokenizer import LatexTokenizer
from torch.utils.data import DataLoader

ROOT_DIR = "../hme100k"
LABEL_FILE = "../hme100k/train.txt"

print("Building tokenizer...")
tokenizer = LatexTokenizer()
labels = []
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            labels.append(parts[1])
tokenizer.build_vocab(labels)
print(f"Vocab size: {tokenizer.vocab_size}")

dataset = HMEDataset(ROOT_DIR, LABEL_FILE, tokenizer, max_samples=50)
print(f"Loaded {len(dataset)} samples")

image, token_ids = dataset[0]
print(f"Image shape: {image.shape}")         
print(f"Token IDs:   {token_ids.tolist()}")
print(f"Decoded:     {tokenizer.decode(token_ids.tolist())}")

loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
images, labels = next(iter(loader))
print(f"Batch images shape: {images.shape}")   
print(f"Batch labels shape: {labels.shape}")  

dataset_aug = HMEDataset(ROOT_DIR, LABEL_FILE, tokenizer,
                         max_samples=10, augment=True)
aug_image, _ = dataset_aug[0]
print(f"Augmented image shape: {aug_image.shape}")
print(f"Pixel range: [{aug_image.min():.2f}, {aug_image.max():.2f}]")

print("\nAll checks passed!")
