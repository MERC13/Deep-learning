# models/temporal_transformer.py
import torch
import torch.nn as nn
from typing import List, Optional


class TemporalEmbedding(nn.Module):
    """
    Build a per-week embedding vector from numerical and categorical features.
    - Numerical features: projected together via a single linear layer to d_model
    - Categorical features: each has its own Embedding(d_model) with a padding index; we average them
    Weekly embedding = num_proj(x_num) + mean(cat_embeds)
    """
    def __init__(self, num_continuous: int, cat_cardinalities_padded: List[int], d_model: int):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities_padded = cat_cardinalities_padded
        self.d_model = d_model

        # Project all numeric features together
        self.num_proj = nn.Linear(num_continuous, d_model)

        # One embedding per categorical feature; padding_idx is the last index
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_model, padding_idx=card-1) for card in cat_cardinalities_padded
        ])

        # LayerNorm helps stabilize when summing numeric and categorical parts
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: (B, T, Cn)
            x_cat: (B, T, Cc) of Long with padding indices for pad time steps
        Returns:
            emb: (B, T, d_model)
        """
        num_emb = self.num_proj(x_num)  # (B, T, d)
        if len(self.cat_cardinalities_padded) == 0:
            emb = self.ln(num_emb)
            return emb

        # Average cat embeddings across features
        cat_vecs = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            # x_cat[..., i] -> (B, T)
            cat_vecs.append(emb_layer(x_cat[..., i]))  # (B, T, d)
        cat_stack = torch.stack(cat_vecs, dim=0)  # (Cc, B, T, d)
        cat_mean = cat_stack.mean(dim=0)  # (B, T, d)

        emb = self.ln(num_emb + cat_mean)  # (B, T, d)
        return emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        T = x.size(1)
        return x + self.pe[:, :T]


class TemporalTransformer(nn.Module):
    """
    Sequence model for week-n prediction using all prior weeks (1..n-1) as input.
    - Embeds each week's tabular features to d_model
    - Applies TransformerEncoder with key padding mask
    - Uses the last valid time step representation to predict week-n fantasy points
    """
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities_padded: List[int],
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.embedder = TemporalEmbedding(num_continuous, cat_cardinalities_padded, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        ff_dim = int(d_model * 4 / 3)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_num: (B, T, Cn)
            x_cat: (B, T, Cc)
            key_padding_mask: (B, T) bool where True indicates padding positions
        Returns:
            (B, 1) predicted fantasy points for week n
        """
        x = self.embedder(x_num, x_cat)  # (B, T, d)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, T, d)

        # Use the last valid time step per sequence (where mask is False)
        if key_padding_mask is None:
            last_repr = x[:, -1, :]
        else:
            lengths = (~key_padding_mask).sum(dim=1)  # (B,)
            # clamp in case of zero-length (shouldn't happen as we enforce at least 1 prior week)
            lengths = lengths.clamp(min=1)
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(-1))  # (B,1,d)
            last_repr = x.gather(dim=1, index=idx).squeeze(1)  # (B, d)

        out = self.head(self.dropout(last_repr))  # (B, 1)
        return out
