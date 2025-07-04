# PatchTST/models/patchtst_univar.py

import torch
import torch.nn as nn

# Univariate PatchTST model
class PatchTST(nn.Module):

    def __init__(
        self,
        in_channels: int,
        seq_len: int = 60,
        patch_size: int = 12,
        emb_dim: int = 64,
        n_heads: int = 4,
        ff_mult: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        channel_independent: bool = True,
        out_horizon: int = 1,
    ):
        super().__init__()
        # Enforce univariate input
        if in_channels != 1:
            raise ValueError("patchtst_univar.py only supports in_channels=1 (univariate input)")
        assert seq_len % patch_size == 0, "seq_len must be a multiple of patch_size"
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.channel_independent = channel_independent

        # Patch embedding: project each patch to embedding space
        if channel_independent:
            # for univariate, this is always a single Conv1d
            self.patch_proj = nn.Conv1d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.patch_proj = nn.Conv1d(1, emb_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head: normalization and output layers
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len)
        # returns: (batch, out_horizon)
        B, C, T = x.shape
        # patchify & embed
        u = self.patch_proj(x)  # (B, emb_dim, n_patches)
        u = u.permute(0, 2, 1) # (B, n_patches, emb_dim)
        u = u + self.pos_embed
        u = self.encoder(u)
        z = u.mean(dim=1)
        z = self.norm(z)
        out = self.head(z)  # (B, out_horizon)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
