import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """
    Standard small MLP predictor head used across baselines and transfer methods.
    input -> Linear -> ReLU -> Dropout -> Linear -> scalar DMS_score
    """

    def __init__(self, in_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RepresentationTransferNetwork(nn.Module):
    """
    Learned representation mapper from source model space to base model space.
    Inspired by SoupFold's pairwise transfer network, adapted here for single-residue mutation embeddings:
    source_dim (768) -> hidden_dim (512) -> base_dim (480)
    """

    def __init__(self, src_dim=768, target_dim=480, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(src_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransferFluorescencePredictor(nn.Module):
    """
    Full representation-transfer prediction pipeline:
    1. Map source representation into ESM space: H_trans = mapper(z_source)
    2. Fuse in latent space: H = (z_esm + alpha * H_trans) / (1 + alpha)
    3. Predict fluorescence: y_hat = regressor(H)
    """

    def __init__(self, mapper, regressor, alpha=1.0, fusion="weighted_avg"):
        super().__init__()
        self.mapper = mapper
        self.regressor = regressor
        self.alpha = alpha
        self.fusion = fusion

    def forward(self, z_esm, z_source):
        h_trans = self.mapper(z_source)
        if self.fusion == "weighted_avg":
            # SoupFold representation-space weighted average
            h = (z_esm + self.alpha * h_trans) / (1.0 + self.alpha)
        elif self.fusion == "residual":
            h = z_esm + self.alpha * h_trans
        elif self.fusion == "concat":
            h = torch.cat([z_esm, h_trans], dim=-1)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        return self.regressor(h)
