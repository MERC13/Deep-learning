# models/ft_transformer.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

class FeatureTokenizer(nn.Module):
    """
    Converts features to embeddings
    - Categorical features: learned embeddings
    - Numerical features: linear projection
    """
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        d_token: int = 192
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        
        # Numerical feature embeddings (one layer per feature)
        self.num_embeddings = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_continuous)
        ])
        
        # Categorical feature embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_token)
            for cardinality in cat_cardinalities
        ])
        
        # CLS token (for final prediction)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
    
    def forward(self, x_num, x_cat):
        """
        Args:
            x_num: (batch_size, num_continuous)
            x_cat: (batch_size, num_categorical)
        Returns:
            tokens: (batch_size, num_features + 1, d_token)
        """
        batch_size = x_num.shape[0]
        
        # Embed numerical features
        num_tokens = []
        for i in range(self.num_continuous):
            token = self.num_embeddings[i](x_num[:, i:i+1])
            num_tokens.append(token)
        num_tokens = torch.stack(num_tokens, dim=1)  # (B, num_cont, d_token)
        
        # Embed categorical features
        cat_tokens = []
        for i in range(len(self.cat_cardinalities)):
            token = self.cat_embeddings[i](x_cat[:, i])
            cat_tokens.append(token)
        cat_tokens = torch.stack(cat_tokens, dim=1)  # (B, num_cat, d_token)
        
        # Concatenate CLS token, numerical tokens, categorical tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, num_tokens, cat_tokens], dim=1)
        
        return tokens


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular data regression
    Predicts fantasy points from game statistics
    """
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn_factor: float = 4/3,
        dropout: float = 0.1,
        prenormalization: bool = True,
        activation: str = 'reglu'
    ):
        super().__init__()
        
        # Feature tokenizer
        self.feature_tokenizer = FeatureTokenizer(
            num_continuous, cat_cardinalities, d_token
        )
        
        # Transformer layers
        d_ffn = int(d_token * d_ffn_factor)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=d_ffn,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=prenormalization
            )
            for _ in range(n_layers)
        ])
        
        # Final prediction head (operates on CLS token)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)  # Regression output
        )
    
    def forward(self, x_num, x_cat):
        """
        Args:
            x_num: (batch_size, num_continuous)
            x_cat: (batch_size, num_categorical)
        Returns:
            predictions: (batch_size, 1)
        """
        # Get feature tokens
        x = self.feature_tokenizer(x_num, x_cat)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Use CLS token for prediction
        cls_token = x[:, 0, :]  # (batch_size, d_token)
        
        # Final prediction
        predictions = self.head(cls_token)
        
        return predictions
