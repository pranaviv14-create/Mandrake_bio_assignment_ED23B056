# GFP Representation Transfer

**Author:** Pranavi Voleti (ED23B056)  
**Task:** Representation transfer for GFP fluorescence prediction under held-out mutation positions.

---

## Overview

This project investigates whether representations from a different protein language model family (**RITA-s**) can improve green fluorescent protein (avGFP) fluorescence prediction when combined with a base **ESM-2 35M** model.

We evaluate models on a **position-held-out split** where mutation positions in the test set were never seen during training.

---

## Models

1. **Base Model:** `facebook/esm2_t12_35M_UR50D` (ESM-2 35M, 12 layers, 480-dimensional hidden states, ~35M parameters, bidirectional masked LM).
2. **Source Model:** `lightonai/RITA_s` (RITA-s, 12 layers, 768-dimensional hidden states, ~85M parameters, autoregressive causal LM).

Both pretrained models are kept frozen. Features are extracted at the specific mutated residue position.

---

## Dataset & Fixed Split

The dataset contains 1,084 single-substitution variants of `GFP_AEQVI_Sarkisyan_2016` (238 amino acids each) from ProteinGym v1.3:

* **Train (`dataset/train.csv`):** 657 variants (Folds 1, 2, 3), assayed positions 50 to 191
* **Validation (`dataset/validation.csv`):** 214 variants (Fold 4), assayed positions 192 to 237
* **Test (`dataset/test.csv`):** 213 variants (Fold 0), assayed positions 3 to 49

No mutated position appears in more than one split. Hyperparameters were selected on the validation set, and the final models were refitted from scratch on all 871 development variants (Train + Validation) before evaluating on the 213 test variants.

---

## Methods Compared

1. **ESM-only:** Frozen ESM-2 residue embedding -> small MLP regressor.
2. **RITA-only:** Frozen RITA-s residue embedding -> small MLP regressor.
3. **ESM + RITA Concatenation:** Simple feature combination [ESM ; RITA] -> small MLP regressor.
4. **Representation Transfer:** 2-layer MLP mapper trained with MSE loss to project RITA representations into ESM latent space, combined with ESM representations via weighted averaging, followed by an MLP regressor.

---

## Results

### Final Test Evaluation (213 Held-Out Variants, Positions 3 to 49)

| Method | Test Spearman (rho) | Test MSE |
| :--- | :---: | :---: |
| **ESM-only** | 0.2175 | 0.4636 |
| **RITA-only** | 0.1817 | 0.4293 |
| **ESM + RITA (Concatenation)** | **0.2764** | **0.4256** |
| **Representation Transfer** | 0.2006 | 0.4405 |

*Observations:*
* The concatenation baseline achieved the highest Spearman correlation (rho = 0.2764), showing that RITA-s carries useful complementary information alongside ESM-2.
* Explicit representation mapping via MSE (rho = 0.2006) did not outperform concatenation in this experiment, indicating that projecting causal representations into the masked model's subspace acted as an information bottleneck.

---

## Repository Structure

```
Mandrake_bio_assignment_ED23B056/
├── GFP_Representation_Transfer.ipynb    # Main runnable Google Colab notebook
├── README.md                            # Project overview
├── report.md                            # Experimental report
├── requirements.txt                     # Dependencies
│
├── dataset/
│   ├── train.csv                        # 657 train variants
│   ├── validation.csv                   # 214 validation variants
│   ├── test.csv                         # 213 test variants
│   ├── reference.fasta                  # Wildtype avGFP reference
│   └── submission_template.csv          # Prediction template
│
├── embeddings/                          # Cached numpy embeddings
│   ├── esm_train.npy
│   ├── esm_validation.npy
│   ├── esm_test.npy
│   ├── source_train.npy
│   ├── source_validation.npy
│   └── source_test.npy
│
└── results/
    ├── submission_esm.csv               # ESM-only predictions
    ├── submission_rita.csv              # RITA-only predictions
    ├── submission_concat.csv            # Concatenation predictions
    ├── submission_transfer.csv          # Transfer predictions
    └── figures/
        ├── dms_distribution.png
        ├── validation_vs_test_spearman.png
        └── test_predictions_scatter.png
```

---

## Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Open the notebook directly from GitHub:
   `https://github.com/pranaviv14-create/Mandrake_bio_assignment_ED23B056/blob/main/GFP_Representation_Transfer.ipynb`
3. Select GPU: **Runtime** -> **Change runtime type** -> **T4 GPU**.
4. Run cells sequentially from top to bottom.
