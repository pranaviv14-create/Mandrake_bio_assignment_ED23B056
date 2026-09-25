# Experimental Results & Artifacts

This directory contains the predictions, summary figures, and artifacts produced by the experimental pipeline.

---

## Prediction Files (`results/predictions/`)

Each file contains predictions for all 213 held-out test variants (Fold 0, positions 3–49) along with ground truth experimental `DMS_score` values:

1. `esm_only_test_preds.csv`: Predictions from the ESM-2 35M baseline head.
2. `source_only_test_preds.csv`: Predictions from the RITA-s baseline head.
3. `concat_test_preds.csv`: Predictions from the concatenated feature model ($[\mathbf{z}_{\text{ESM}} \,\|\, \mathbf{z}_{\text{RITA}}]$).
4. `transfer_test_preds.csv`: Predictions from the SoupFold-inspired representation transfer model.

Each CSV follows the format:
```csv
variant_id,mutant,predicted_fitness,DMS_score
```

---

## Visualizations (`results/figures/`)

1. `dms_distribution.png`: Histogram displaying the distribution of continuous DMS fluorescence measurements across the Train (657), Validation (214), and Test (213) partitions.
2. `validation_vs_test_spearman.png`: Grouped bar chart comparing the primary evaluation metric (Spearman rank correlation $\rho$) between validation (positions 192–237) and held-out test (positions 3–49) across all evaluated methods.
3. `test_predictions_scatter.png`: Two-panel scatter plot of actual vs. predicted fluorescence scores for the ESM-only baseline and the Representation Transfer model, showing correlation and calibration lines.

---

## Summary Performance Table

| Method | Validation Spearman $\rho$ | Validation MSE | Test Spearman $\rho$ | Test MSE |
| :--- | :---: | :---: | :---: | :---: |
| **ESM-only (Baseline 1)** | 0.2027 | 0.3908 | 0.2175 | 0.4636 |
| **Source-only RITA (Baseline 2)** | 0.1651 | 0.3979 | 0.1817 | 0.4293 |
| **Concatenation (Baseline 3)** | **0.2408** | **0.3069** | **0.2764** | **0.4256** |
| **Representation Transfer** | 0.1720 | 0.3537 | 0.2006 | 0.4405 |
