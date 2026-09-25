# GFP Representation Transfer

**Author:** Pranavi Voleti (ED23B056)

This project investigates whether representations from an architecturally distinct protein language model (**RITA-s**) can improve green fluorescent protein (avGFP) fluorescence prediction when combined with a base **ESM-2 35M** encoder. Models are evaluated under a strict position-held-out split where mutations in the test set occur at residue positions never seen during training.

---

## Dataset

- **Dataset:** avGFP Single Mutants from Sarkisyan et al. (2016) / ProteinGym v1.3.
- **Total Variants:** 1,084 single substitutions (length 238 amino acids each).
- **Official Split:**
  - **Train:** 657 variants (positions 50 to 191)
  - **Validation:** 214 variants (positions 192 to 237)
  - **Test:** 213 variants (positions 3 to 49)
  - **Train + Validation (Refit):** 871 variants (positions 50 to 237)
- Mutated positions are strictly disjoint across splits with zero leakage.

---

## Models

- **Base Model:** ESM-2 35M (`facebook/esm2_t12_35M_UR50D`, 480 hidden dim, ~35M parameters, bidirectional MLM).
- **Source Model:** RITA-s (`lightonai/RITA_s`, 768 hidden dim, ~85M parameters, autoregressive causal LM).
- Both pretrained encoders are frozen (`requires_grad = False`, `.eval()`).
- Features are extracted as residue-level embeddings at the mutated position.

---

## Experiments

1. **ESM-only:** Frozen ESM-2 residue embedding (480-dim) -> MLP regressor.
2. **RITA-only:** Frozen RITA-s residue embedding (768-dim) -> MLP regressor.
3. **ESM + RITA:** Direct feature concatenation `[ESM ; RITA]` (1,248-dim) -> MLP regressor.
4. **Representation Transfer:** 2-layer MLP mapper projecting RITA representations into ESM latent space, combined via weighted average ($H = \frac{z_{\text{ESM}} + 0.5 \cdot f(z_{\text{RITA}})}{1.5}$), followed by an MLP regressor.

---

## Results

Final evaluation on the 213 held-out test variants after refitting on all 871 development variants:

| Method | Test Spearman ($\rho$) | Test MSE |
| :--- | :---: | :---: |
| **ESM-only** | 0.217509 | 0.463568 |
| **RITA-only** | 0.181734 | 0.429295 |
| **ESM + RITA (Concatenation)** | **0.276425** | **0.425582** |
| **Representation Transfer** | 0.200646 | 0.440457 |

---

## Running the Notebook

### Google Colab
1. Open [`GFP_Representation_Transfer.ipynb`](GFP_Representation_Transfer.ipynb) in Google Colab.
2. Select a GPU runtime: **Runtime** -> **Change runtime type** -> **T4 GPU** (or run on CPU).
3. If running without cloning beforehand, the notebook automatically clones the repository and sets up the working directory if `dataset/` is not present.
4. Run all cells sequentially. Pre-extracted embeddings in `embeddings/` allow the entire notebook to run in under 1 minute.

### Local Setup
```bash
git clone https://github.com/pranaviv14-create/Mandrake_bio_assignment_ED23B056.git
cd Mandrake_bio_assignment_ED23B056
pip install -r requirements.txt
jupyter notebook GFP_Representation_Transfer.ipynb
```

---

## Report

For detailed experimental design, validation results, refit protocol, scientific discussion, and limitations, refer to:

[`report.md`](report.md)
