import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Ensure root is in pythonpath
base_dir = r"C:\Users\Dell\.gemini\antigravity\scratch\gfp-representation-transfer"
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_utils import load_dataset, verify_split_leakage
from src.embeddings import get_or_create_cached_embeddings
from src.models import MLPPredictor, RepresentationTransferNetwork, TransferFluorescencePredictor
from src.evaluation import evaluate_predictions, format_results_table
from src.train import set_seed, train_regressor, train_representation_mapper

print("=" * 60)
print("GFP Fluorescence Prediction: Representation Transfer Pipeline")
print("=" * 60)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Data
dataset_dir = os.path.join(base_dir, "dataset")
df_train, df_val, df_test, ref_seq = load_dataset(dataset_dir)
print(f"Data Loaded: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
print(f"Reference sequence length: {len(ref_seq)}")

# Verify zero leakage
split_stats = verify_split_leakage(df_train, df_val, df_test)
print("Leakage verification passed: No positional overlap across splits!")
print(f"Train positions: {split_stats['train_positions']}")
print(f"Val positions:   {split_stats['val_positions']}")
print(f"Test positions:  {split_stats['test_positions']}")

# 2. Extract / Load Cached Embeddings
cache_dir = os.path.join(base_dir, "embeddings")
embs = get_or_create_cached_embeddings(df_train, df_val, df_test, cache_dir, device=device, batch_size=16)

z_esm_tr = embs["esm_train"]
z_esm_val = embs["esm_val"]
z_esm_test = embs["esm_test"]

z_src_tr = embs["source_train"]
z_src_val = embs["source_val"]
z_src_test = embs["source_test"]

y_train = df_train["DMS_score"].values.astype(np.float32)
y_val = df_val["DMS_score"].values.astype(np.float32)
y_test = df_test["DMS_score"].values.astype(np.float32)

print("\nEmbedding shapes:")
print(f"ESM Train: {z_esm_tr.shape}, Val: {z_esm_val.shape}, Test: {z_esm_test.shape}")
print(f"Source Train: {z_src_tr.shape}, Val: {z_src_val.shape}, Test: {z_src_test.shape}")

# Scaler fit ONLY on training set
mean_esm = z_esm_tr.mean(axis=0, keepdims=True)
std_esm = z_esm_tr.std(axis=0, keepdims=True) + 1e-6

mean_src = z_src_tr.mean(axis=0, keepdims=True)
std_src = z_src_tr.std(axis=0, keepdims=True) + 1e-6

z_esm_tr_norm = (z_esm_tr - mean_esm) / std_esm
z_esm_val_norm = (z_esm_val - mean_esm) / std_esm
z_esm_test_norm = (z_esm_test - mean_esm) / std_esm

z_src_tr_norm = (z_src_tr - mean_src) / std_src
z_src_val_norm = (z_src_val - mean_src) / std_src
z_src_test_norm = (z_src_test - mean_src) / std_src

# Combined normalized features for concatenation baseline
z_cat_tr = np.concatenate([z_esm_tr_norm, z_src_tr_norm], axis=1)
z_cat_val = np.concatenate([z_esm_val_norm, z_src_val_norm], axis=1)
z_cat_test = np.concatenate([z_esm_test_norm, z_src_test_norm], axis=1)

# Target normalization based ONLY on training set
y_mean = float(y_train.mean())
y_std = float(y_train.std()) + 1e-6
y_tr_norm = (y_train - y_mean) / y_std
y_val_norm = (y_val - y_mean) / y_std

# 3. Model Development & Validation
val_results = {}
val_preds_dict = {}
epochs_recorded = {}

# BASELINE 1: ESM-only
print("\n--- Training Baseline 1: ESM-only ---")
esm_model, esm_epochs = train_regressor(
    z_esm_tr_norm, y_tr_norm, z_esm_val_norm, y_val_norm,
    hidden_dim=128, lr=1e-3, epochs=60, patience=10, device=device
)
with torch.no_grad():
    p_esm_val = esm_model(torch.from_numpy(z_esm_val_norm).float().to(device)).cpu().numpy()
p_esm_val_orig = p_esm_val * y_std + y_mean
val_results["ESM-only (Baseline 1)"] = evaluate_predictions(y_val, p_esm_val_orig)
val_preds_dict["ESM-only (Baseline 1)"] = p_esm_val_orig
epochs_recorded["ESM-only"] = max(esm_epochs, 20)

# BASELINE 2: Source-only (RITA-s)
print("\n--- Training Baseline 2: Source-only (RITA-s) ---")
src_model, src_epochs = train_regressor(
    z_src_tr_norm, y_tr_norm, z_src_val_norm, y_val_norm,
    hidden_dim=128, lr=1e-3, epochs=60, patience=10, device=device
)
with torch.no_grad():
    p_src_val = src_model(torch.from_numpy(z_src_val_norm).float().to(device)).cpu().numpy()
p_src_val_orig = p_src_val * y_std + y_mean
val_results["Source-only RITA (Baseline 2)"] = evaluate_predictions(y_val, p_src_val_orig)
val_preds_dict["Source-only RITA (Baseline 2)"] = p_src_val_orig
epochs_recorded["Source-only"] = max(src_epochs, 20)

# BASELINE 3: Feature Concatenation
print("\n--- Training Baseline 3: Simple Concatenation ---")
cat_model, cat_epochs = train_regressor(
    z_cat_tr, y_tr_norm, z_cat_val, y_val_norm,
    hidden_dim=128, lr=1e-3, epochs=60, patience=10, device=device
)
with torch.no_grad():
    p_cat_val = cat_model(torch.from_numpy(z_cat_val).float().to(device)).cpu().numpy()
p_cat_val_orig = p_cat_val * y_std + y_mean
val_results["Concatenation (Baseline 3)"] = evaluate_predictions(y_val, p_cat_val_orig)
val_preds_dict["Concatenation (Baseline 3)"] = p_cat_val_orig
epochs_recorded["Concatenation"] = max(cat_epochs, 20)

# METHOD 4: Learned Representation Transfer (SoupFold-inspired)
print("\n--- Training Method 4: Learned Representation Transfer ---")
# Step A: Train transfer mapper ft->b on TRAIN representations only
mapper = train_representation_mapper(
    z_src_tr_norm, z_esm_tr_norm,
    src_dim=768, target_dim=480, hidden_dim=512,
    lr=1e-3, epochs=40, batch_size=32, device=device
)
mapper.eval()
with torch.no_grad():
    mapped_src_tr = mapper(torch.from_numpy(z_src_tr_norm).float().to(device)).cpu().numpy()
    mapped_src_val = mapper(torch.from_numpy(z_src_val_norm).float().to(device)).cpu().numpy()
    mapped_src_test = mapper(torch.from_numpy(z_src_test_norm).float().to(device)).cpu().numpy()

# Fusion: SoupFold latent weighted average (wb = wt = 1 -> alpha = 1.0)
alpha = 0.5  # blending parameter
H_tr = (z_esm_tr_norm + alpha * mapped_src_tr) / (1.0 + alpha)
H_val = (z_esm_val_norm + alpha * mapped_src_val) / (1.0 + alpha)
H_test = (z_esm_test_norm + alpha * mapped_src_test) / (1.0 + alpha)

trans_model, trans_epochs = train_regressor(
    H_tr, y_tr_norm, H_val, y_val_norm,
    hidden_dim=128, lr=1e-3, epochs=60, patience=10, device=device
)
with torch.no_grad():
    p_trans_val = trans_model(torch.from_numpy(H_val).float().to(device)).cpu().numpy()
p_trans_val_orig = p_trans_val * y_std + y_mean
val_results["Representation Transfer (SoupFold-inspired)"] = evaluate_predictions(y_val, p_trans_val_orig)
val_preds_dict["Representation Transfer (SoupFold-inspired)"] = p_trans_val_orig
epochs_recorded["Representation Transfer"] = max(trans_epochs, 20)

print("\n" + format_results_table(val_results, "Validation"))

# 4. Final Refit on Combined Train + Validation (871 samples)
print("\n" + "=" * 60)
print("Final Refit from scratch on Train + Validation (871 samples)")
print("=" * 60)

# Combine Train + Val
df_dev = pd.concat([df_train, df_val], ignore_index=True)
y_dev = df_dev["DMS_score"].values.astype(np.float32)

z_esm_dev = np.concatenate([z_esm_tr, z_esm_val], axis=0)
z_src_dev = np.concatenate([z_src_tr, z_src_val], axis=0)

# Re-fit scalers on all 871 dev samples
mean_esm_dev = z_esm_dev.mean(axis=0, keepdims=True)
std_esm_dev = z_esm_dev.std(axis=0, keepdims=True) + 1e-6

mean_src_dev = z_src_dev.mean(axis=0, keepdims=True)
std_src_dev = z_src_dev.std(axis=0, keepdims=True) + 1e-6

z_esm_dev_norm = (z_esm_dev - mean_esm_dev) / std_esm_dev
z_esm_test_refit = (z_esm_test - mean_esm_dev) / std_esm_dev

z_src_dev_norm = (z_src_dev - mean_src_dev) / std_src_dev
z_src_test_refit = (z_src_test - mean_src_dev) / std_src_dev

z_cat_dev = np.concatenate([z_esm_dev_norm, z_src_dev_norm], axis=1)
z_cat_test_refit = np.concatenate([z_esm_test_refit, z_src_test_refit], axis=1)

y_mean_dev = float(y_dev.mean())
y_std_dev = float(y_dev.std()) + 1e-6
y_dev_norm = (y_dev - y_mean_dev) / y_std_dev

# Refit Baseline 1: ESM-only
set_seed(42)
esm_refit, _ = train_regressor(
    z_esm_dev_norm, y_dev_norm, x_val=None, y_val=None,
    hidden_dim=128, lr=1e-3, epochs=epochs_recorded["ESM-only"], device=device
)
with torch.no_grad():
    pred_esm_test = esm_refit(torch.from_numpy(z_esm_test_refit).float().to(device)).cpu().numpy()
pred_esm_test_orig = pred_esm_test * y_std_dev + y_mean_dev

# Refit Baseline 2: Source-only
set_seed(42)
src_refit, _ = train_regressor(
    z_src_dev_norm, y_dev_norm, x_val=None, y_val=None,
    hidden_dim=128, lr=1e-3, epochs=epochs_recorded["Source-only"], device=device
)
with torch.no_grad():
    pred_src_test = src_refit(torch.from_numpy(z_src_test_refit).float().to(device)).cpu().numpy()
pred_src_test_orig = pred_src_test * y_std_dev + y_mean_dev

# Refit Baseline 3: Concatenation
set_seed(42)
cat_refit, _ = train_regressor(
    z_cat_dev, y_dev_norm, x_val=None, y_val=None,
    hidden_dim=128, lr=1e-3, epochs=epochs_recorded["Concatenation"], device=device
)
with torch.no_grad():
    pred_cat_test = cat_refit(torch.from_numpy(z_cat_test_refit).float().to(device)).cpu().numpy()
pred_cat_test_orig = pred_cat_test * y_std_dev + y_mean_dev

# Refit Method 4: Representation Transfer
set_seed(42)
mapper_refit = train_representation_mapper(
    z_src_dev_norm, z_esm_dev_norm,
    src_dim=768, target_dim=480, hidden_dim=512,
    lr=1e-3, epochs=40, batch_size=32, device=device
)
mapper_refit.eval()
with torch.no_grad():
    mapped_src_dev = mapper_refit(torch.from_numpy(z_src_dev_norm).float().to(device)).cpu().numpy()
    mapped_src_test_refit = mapper_refit(torch.from_numpy(z_src_test_refit).float().to(device)).cpu().numpy()

H_dev = (z_esm_dev_norm + alpha * mapped_src_dev) / (1.0 + alpha)
H_test_refit = (z_esm_test_refit + alpha * mapped_src_test_refit) / (1.0 + alpha)

set_seed(42)
trans_refit, _ = train_regressor(
    H_dev, y_dev_norm, x_val=None, y_val=None,
    hidden_dim=128, lr=1e-3, epochs=epochs_recorded["Representation Transfer"], device=device
)
with torch.no_grad():
    pred_trans_test = trans_refit(torch.from_numpy(H_test_refit).float().to(device)).cpu().numpy()
pred_trans_test_orig = pred_trans_test * y_std_dev + y_mean_dev

# 5. Final Evaluation on Held-Out Test Variants (213 samples)
print("\n" + "=" * 60)
print("Final Evaluation on Held-Out Test Variants (Fold 0, Mutated Positions 3-49)")
print("=" * 60)

test_results = {
    "ESM-only (Baseline 1)": evaluate_predictions(y_test, pred_esm_test_orig),
    "Source-only RITA (Baseline 2)": evaluate_predictions(y_test, pred_src_test_orig),
    "Concatenation (Baseline 3)": evaluate_predictions(y_test, pred_cat_test_orig),
    "Representation Transfer (SoupFold-inspired)": evaluate_predictions(y_test, pred_trans_test_orig),
}

print(format_results_table(test_results, "Final Test (Held-out Positions)"))

# 6. Save Predictions to results/predictions
results_dir = os.path.join(base_dir, "results")
preds_dir = os.path.join(results_dir, "predictions")
figs_dir = os.path.join(results_dir, "figures")
os.makedirs(preds_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)

for name, preds in [
    ("esm_only_test_preds.csv", pred_esm_test_orig),
    ("source_only_test_preds.csv", pred_src_test_orig),
    ("concat_test_preds.csv", pred_cat_test_orig),
    ("transfer_test_preds.csv", pred_trans_test_orig),
]:
    pred_df = pd.DataFrame({
        "variant_id": df_test["variant_id"],
        "mutant": df_test["mutant"],
        "predicted_fitness": preds,
        "DMS_score": y_test,
    })
    pred_df.to_csv(os.path.join(preds_dir, name), index=False)
print("Saved all test prediction CSVs to results/predictions/")

# 7. Generate Visualizations
# Plot 1: DMS_score distribution
plt.figure(figsize=(8, 4))
plt.hist(df_train["DMS_score"], bins=30, alpha=0.6, label=f"Train (N={len(df_train)})", color="steelblue")
plt.hist(df_val["DMS_score"], bins=30, alpha=0.6, label=f"Validation (N={len(df_val)})", color="orange")
plt.hist(df_test["DMS_score"], bins=30, alpha=0.6, label=f"Test (N={len(df_test)})", color="green")
plt.xlabel("DMS Score (Fluorescence)")
plt.ylabel("Variant Count")
plt.title("GFP DMS Score Distribution across Official Splits")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, "dms_distribution.png"), dpi=150)
plt.close()

# Plot 2: Validation vs Test Spearman Comparison
methods = list(val_results.keys())
labels = ["ESM-only", "Source (RITA)", "Concatenation", "Transfer"]
val_rhos = [val_results[m]["spearman"] for m in methods]
test_rhos = [test_results[m]["spearman"] for m in methods]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 4.5))
plt.bar(x - width/2, val_rhos, width, label="Validation (Pos 192-237)", color="royalblue", alpha=0.85)
plt.bar(x + width/2, test_rhos, width, label="Test (Pos 3-49)", color="darkseagreen", alpha=0.85)
plt.ylabel("Spearman Rank Correlation (ρ)")
plt.title("Performance Comparison: Baselines vs Representation Transfer")
plt.xticks(x, labels)
plt.ylim(0.0, max(max(val_rhos), max(test_rhos)) + 0.15)
plt.legend()
for i in range(len(labels)):
    plt.text(x[i] - width/2, val_rhos[i] + 0.01, f"{val_rhos[i]:.3f}", ha="center", fontsize=9)
    plt.text(x[i] + width/2, test_rhos[i] + 0.01, f"{test_rhos[i]:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, "validation_vs_test_spearman.png"), dpi=150)
plt.close()

# Plot 3: Scatter of Predicted vs Actual Test Scores
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

axes[0].scatter(y_test, pred_esm_test_orig, alpha=0.6, edgecolors="none", color="steelblue", s=25)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1)
axes[0].set_title(f"Baseline 1: ESM-only\nSpearman ρ = {test_results['ESM-only (Baseline 1)']['spearman']:.3f}, MSE = {test_results['ESM-only (Baseline 1)']['mse']:.3f}")
axes[0].set_xlabel("Actual DMS Score")
axes[0].set_ylabel("Predicted Fitness")

axes[1].scatter(y_test, pred_trans_test_orig, alpha=0.6, edgecolors="none", color="seagreen", s=25)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1)
axes[1].set_title(f"Representation Transfer\nSpearman ρ = {test_results['Representation Transfer (SoupFold-inspired)']['spearman']:.3f}, MSE = {test_results['Representation Transfer (SoupFold-inspired)']['mse']:.3f}")
axes[1].set_xlabel("Actual DMS Score")

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, "test_predictions_scatter.png"), dpi=150)
plt.close()

print("Figures successfully generated in results/figures/")
