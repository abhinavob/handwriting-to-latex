from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HMEDataset(Dataset):
    def __init__(self, root_dir, label_file, tokenizer=None, max_samples=100):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.tokenizer = tokenizer

        with open(label_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                img_path, label = line.strip().split('\t')
                self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]

        img_path = self.root_dir / img_rel_path
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        tokens = self.tokenizer.encode(label)

        return image, tokens
