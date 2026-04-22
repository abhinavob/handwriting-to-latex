import math
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
# Change this at the top:
from torchvision.models import resnet34, ResNet34_Weights

# ------------------------------------------------------------
# 2D POSITIONAL ENCODING (Based on Im2LaTeX-100k paper)
# ------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        self.d_model = d_model
        
        # We split the dimensions in half: 256 for X, 256 for Y
        d_pos = d_model // 2
        
        pe = torch.zeros(d_model, max_h, max_w)
        
        # Denominator for the frequencies
        div_term = torch.exp(torch.arange(0, d_pos, 2).float() * (-math.log(10000.0) / d_pos))
        
        # 1. Horizontal (X) encoding - applied across the width
        pos_w = torch.arange(0, max_w).float().unsqueeze(1) # [max_w, 1]
        pe_w = torch.zeros(max_w, d_pos)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)
        
        # Transpose and expand to match image height: [d_pos, max_h, max_w]
        pe_w = pe_w.transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)
        pe[:d_pos, :, :] = pe_w
        
        # 2. Vertical (Y) encoding - applied across the height
        pos_h = torch.arange(0, max_h).float().unsqueeze(1) # [max_h, 1]
        pe_h = torch.zeros(max_h, d_pos)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)
        
        # Transpose and expand to match image width: [d_pos, max_h, max_w]
        pe_h = pe_h.transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        pe[d_pos:, :, :] = pe_h
        
        # Shape is [d_model, max_h, max_w]. Add batch dim -> [1, d_model, max_h, max_w]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [B, d_model, H, W]
        H, W = x.size(2), x.size(3)
        
        # Add the 2D positional encoding to the feature map
        return x + self.pe[:, :, :H, :W]


# ------------------------------------------------------------
# ENCODER
# ------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet backbone
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        # We strip off the final Average Pool and Linear layer so we get the spatial grid
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Initialize the 2D Positional Encoding
        self.pos_encoding2d = PositionalEncoding2D(d_model=512)

    def forward(self, x):
        # x: [B, 3, 224, 224]

        # 1. Extract visual features from the CNN
        features = self.backbone(x)             # [B, 512, 7, 7]
        
        # 2. Add 2D spatial context BEFORE flattening
        features = self.pos_encoding2d(features) 

        # 3. Flatten for the Transformer Decoder
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)   # [B, 512, 49]
        features = features.permute(0, 2, 1)    # [B, 49, 512]

        # Calculate lengths for the padding mask
        src_len = features.size(1)
        source_lengths = torch.full((B,), src_len, dtype=torch.long, device=x.device)

        return features, source_lengths
