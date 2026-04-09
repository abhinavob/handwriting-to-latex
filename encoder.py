import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from dataset import HMEDataset

# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# Encoder

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet backbone
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=512)

    def forward(self, x):
        # x: [B, 3, 224, 224]

        features = self.backbone(x)          # [B, 512, 7, 7]

        B, C, H, W = features.shape

        # Flatten
        features = features.view(B, C, H * W)   # [B, 512, 49]
        features = features.permute(0, 2, 1)    # [B, 49, 512]

        # Add positional encoding
        features = self.pos_encoding(features)  # [B, 49, 512]
        B = features.size(0)
        src_len = features.size(1)
        source_lengths = torch.full((B,), src_len, dtype=torch.long, device=x.device)

        return features, source_lengths
