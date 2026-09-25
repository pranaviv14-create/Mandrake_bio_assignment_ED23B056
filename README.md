# GFP Representation Transfer

This repository contains the complete experimental code, notebook, datasets, and report for investigating **representation transfer for GFP fluorescence prediction** across position-held-out splits.

---

## Objective

The goal of this project is to investigate whether information from an architecturally distinct protein language model (**RITA-s**, an autoregressive causal LM) can improve fluorescence prediction from a base **ESM-2 35M** model on single-substitution variants at mutation positions completely held out from development data.

---

## Dataset

* **Protein:** avGFP (`GFP_AEQVI_Sarkisyan_2016`) from ProteinGym v1.3.
* **Total Variants:** 1,084 single amino-acid substitutions across 233 distinct mutated positions.
* **Target:** Continuous fluorescence measurement (`DMS_score`).
* **Fixed Position-Held-Out Split:**
  * **Train (`dataset/train.csv`):** 657 variants (Folds 1, 2, 3), mutated positions 50–191.
  * **Validation (`dataset/validation.csv`):** 214 variants (Fold 4), mutated positions 192–237.
  * **Test (`dataset/test.csv`):** 213 variants (Fold 0), mutated positions 3–49.

> **Strict Protocol:** Mutated positions never overlap across train, validation, and test sets. Pretrained encoders remain frozen. All models are selected on validation and refitted from scratch on all 871 development variants (Train + Val) before final test evaluation.

---

## Models

1. **Base Model:** **ESM-2 35M** (`facebook/esm2_t12_35M_UR50D`) — 12-layer bidirectional masked LM ($d=480$).
2. **Source Model:** **RITA-s** (`lightonai/RITA_s`) — 12-layer autoregressive causal LM ($d=768$).

---

## Methods Compared

1. **Baseline 1: ESM-only:** Frozen ESM-2 35M residue embedding $\to$ MLP regressor.
2. **Baseline 2: Source-only:** Frozen RITA-s residue embedding $\to$ MLP regressor.
3. **Baseline 3: Concatenation:** Concatenated $[\mathbf{z}_{\text{ESM}} \,\|\, \mathbf{z}_{\text{RITA}}] \to$ MLP regressor.
4. **Method 4: Representation Transfer (SoupFold-Inspired):** 2-layer MLP mapper trained with MSE loss to project RITA-s embeddings into ESM-2 latent space, followed by latent weighted averaging and fluorescence regression.

---

## Summary of Results

### Validation (Mutated Positions 192–237)
| Method | Spearman $\rho$ | MSE |
| :--- | :---: | :---: |
| ESM-only (Baseline 1) | 0.2027 | 0.3908 |
| Source-only RITA (Baseline 2) | 0.1651 | 0.3979 |
| **Concatenation (Baseline 3)** | **0.2408** | **0.3069** |
| Representation Transfer | 0.1720 | 0.3537 |

### Final Test Evaluation (Mutated Positions 3–49, Refit on 871 Variants)
| Method | Spearman $\rho$ | MSE |
| :--- | :---: | :---: |
| ESM-only (Baseline 1) | 0.2175 | 0.4636 |
| Source-only RITA (Baseline 2) | 0.1817 | 0.4293 |
| **Concatenation (Baseline 3)** | **0.2764** | **0.4256** |
| Representation Transfer | 0.2006 | 0.4405 |

*Key Takeaway:* Combining representations from the two model families via feature concatenation provides a **+27.1% relative improvement in Spearman correlation** on held-out test positions, confirming that autoregressive protein models carry complementary information for fitness ranking.

---

## Repository Structure

```
gfp-representation-transfer/
│
├── GFP_Representation_Transfer.ipynb    # Main end-to-end Google Colab notebook
├── README.md                            # Project overview & reproduction guide
├── report.md                            # Comprehensive scientific report
├── requirements.txt                     # Dependencies
├── .gitignore                           # Git ignore rules
│
├── dataset/
│   ├── train.csv                        # 657 variants (positions 50–191)
│   ├── validation.csv                   # 214 variants (positions 192–237)
│   ├── test.csv                         # 213 variants (positions 3–49)
│   ├── reference.fasta                  # Wildtype avGFP reference sequence (238 aa)
│   └── submission_template.csv          # Blank test prediction template
│
├── embeddings/                          # Cached numpy embeddings (.npy)
│   ├── esm_train.npy
│   ├── esm_validation.npy
│   ├── esm_test.npy
│   ├── source_train.npy
│   ├── source_validation.npy
│   └── source_test.npy
│
├── src/                                 # Modular helper code
│   ├── __init__.py
│   ├── data_utils.py                    # Data loading and leakage checks
│   ├── embeddings.py                    # ESM-2 and RITA embedding extraction & token alignment
│   ├── models.py                        # Transfer networks and prediction heads
│   ├── train.py                         # Training loops, early stopping, and refitting
│   └── evaluation.py                    # Spearman correlation, MSE, and tables
│
└── results/
    ├── README.md                        # Description of outputs
    ├── figures/                         # Figures and plots
    │   ├── dms_distribution.png
    │   ├── validation_vs_test_spearman.png
    │   └── test_predictions_scatter.png
    └── predictions/                     # CSV prediction files for each method
        ├── esm_only_test_preds.csv
        ├── source_only_test_preds.csv
        ├── concat_test_preds.csv
        └── transfer_test_preds.csv
```

---

## Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Upload `GFP_Representation_Transfer.ipynb` (or open directly from GitHub).
3. Set runtime to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
4. Ensure the repository files or `dataset/` directory are available in your working environment.
5. Run all cells sequentially.

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/gfp-representation-transfer.git
cd gfp-representation-transfer

# Create virtual environment and install requirements
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline
python run_experiments.py
```
