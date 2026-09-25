import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation import evaluate_predictions
from src.models import (
    MLPPredictor,
    RepresentationTransferNetwork,
    TransferFluorescencePredictor,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_representation_mapper(
    z_source_train,
    z_esm_train,
    src_dim=768,
    target_dim=480,
    hidden_dim=512,
    lr=1e-3,
    epochs=50,
    batch_size=32,
    device="cpu",
):
    """
    Train a 2-layer MLP to map source embeddings into ESM space using MSE loss on training data only.
    """
    mapper = RepresentationTransferNetwork(src_dim, target_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(z_source_train).float(),
        torch.from_numpy(z_esm_train).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mapper.train()
    for epoch in range(epochs):
        for x_src, y_esm in loader:
            x_src = x_src.to(device)
            y_esm = y_esm.to(device)

            optimizer.zero_grad()
            pred = mapper(x_src)
            loss = criterion(pred, y_esm)
            loss.backward()
            optimizer.step()

    mapper.eval()
    return mapper


def train_regressor(
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    hidden_dim=128,
    lr=1e-3,
    epochs=60,
    batch_size=32,
    patience=10,
    device="cpu",
):
    """
    Train MLP regressor with early stopping based on validation Spearman rank correlation.
    If x_val is None (e.g. final refit on train+val), train for fixed number of epochs without early stopping.
    """
    in_dim = x_train.shape[1]
    model = MLPPredictor(in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_spearman = -1.0
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(torch.from_numpy(x_val).float().to(device)).cpu().numpy()
            metrics = evaluate_predictions(y_val, val_preds)
            val_spearman = metrics["spearman"]

            if val_spearman > best_spearman:
                best_spearman = val_spearman
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    return model, best_epoch
