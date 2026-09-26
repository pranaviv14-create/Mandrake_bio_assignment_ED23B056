# GFP Representation Transfer

**Author:** Pranavi Voleti (ED23B056)

---

## 1. Objective

The goal of this assignment is to investigate whether transferring representations from an architecturally distinct protein language model can improve GFP fluorescence prediction when ESM-2 is used as the base representation.

We evaluate models on the green fluorescent protein (avGFP) single-substitution dataset from Sarkisyan et al. (2016) / ProteinGym v1.3. Predictions target experimental fluorescence (`DMS_score`) under a strict position-held-out split where mutations in the test set occur at residue positions never seen during training.

---

## 2. Dataset and Split

The dataset contains 1,084 single-substitution variants of avGFP (238 amino acids each). We follow the official ProteinGym contiguous fold split:

- **Train:** 657 variants (Folds 1, 2, 3), assayed positions 50 to 191
- **Validation:** 214 variants (Fold 4), assayed positions 192 to 237
- **Test:** 213 variants (Fold 0), assayed positions 3 to 49
- **Train + Validation for final refit:** 871 variants (Folds 1 to 4), assayed positions 50 to 237

Mutation positions are strictly disjoint across splits: no mutated position in the test set appears in either the training or validation sets. No random splits are introduced.

---

## 3. Models and Representations

1. **Base Model: ESM-2 35M** (`facebook/esm2_t12_35M_UR50D`)
   - 12 transformer layers, 480 hidden dimensions, approximately 35M parameters.
   - Pretrained using masked language modeling (bidirectional context) on UniRef50.
   - Prepending a `<cls>` token maps 1-based residue position $p$ to token index $p$.
   - ESM representation size: 480.

2. **Source Model: RITA-s** (`lightonai/RITA_s`)
   - 12 causal transformer layers, 768 hidden dimensions, approximately 85M parameters.
   - Pretrained using autoregressive next-token prediction on UniRef100.
   - No prepended start token maps 1-based residue position $p$ to token index $p - 1$.
   - RITA representation size: 768.

Both pretrained encoders remain strictly frozen (`requires_grad = False`, `.eval()`). Features are extracted as residue-level embeddings at the specific mutated residue position.

---

## 4. Experiments

We compare four approaches using identical prediction head architectures (`Linear(in_dim, 128) -> ReLU() -> Dropout(0.1) -> Linear(128, 1)`):

1. **ESM-only:** Frozen ESM-2 residue embedding (480 dimensions) passed to the prediction head.
2. **RITA-only:** Frozen RITA-s residue embedding (768 dimensions) passed to the prediction head.
3. **ESM + RITA concatenation:** Direct feature concatenation `[ESM ; RITA]` (1,248 dimensions) passed to the prediction head.
4. **Representation transfer:**
   - **Mapper architecture:** A 2-layer MLP (`Linear(768, 512) -> ReLU() -> Linear(512, 480)`).
   - **Training:** Trained with MSE loss strictly on the 657 training variants to project RITA representations into the ESM latent space.
   - **Latent combination:** Mapped RITA representations are combined with base ESM representations via weighted averaging:
     $$H = \frac{z_{\text{ESM}} + \alpha \cdot f(z_{\text{RITA}})}{1 + \alpha} \quad (\alpha = 0.5)$$
     Setting $\alpha = 0.5$ weights the base ESM representation twice as heavily as the transferred RITA representation.
   - **Validation:** Validation data was used exclusively for method and hyperparameter selection.

---

## 5. Validation Results

Validation performance on the 214 validation variants (mutated positions 192 to 237):

| Method | Validation Spearman ($\rho$) | Validation MSE |
| :--- | :---: | :---: |
| **ESM-only** | 0.202739 | 0.390827 |
| **RITA-only** | 0.165058 | 0.397948 |
| **ESM + RITA (Concatenation)** | **0.240830** | **0.306935** |
| **Representation Transfer** | 0.172027 | 0.353684 |

---

## 6. Final Refit and Test Results

The number of training epochs for each predictor was selected using validation performance with early stopping. The selected epoch count was then fixed before the final refit on the combined training and validation data. After fixing model architectures and settings, each approach was reinitialized and refit from scratch using all 871 development variants (Train + Validation). Final evaluation was then performed once on the 213 held-out test variants (mutated positions 3 to 49):

| Method | Test Spearman ($\rho$) | Test MSE |
| :--- | :---: | :---: |
| **ESM-only** | 0.217618 | 0.463568 |
| **RITA-only** | 0.156970 | 0.438006 |
| **ESM + RITA (Concatenation)** | **0.337409** | **0.349592** |
| **Representation Transfer** | 0.200757 | 0.440457 |

The ESM + RITA concatenation model obtained a test Spearman correlation of 0.337409, compared with 0.217618 for ESM-only. The representation-transfer model achieved a test Spearman correlation of 0.200757.

---

## 7. Interpretation

- Combining the two frozen representations via concatenation improved the validation and test metrics relative to ESM-only in this experiment, suggesting that the RITA-s representation contains information that was useful in combination with ESM-2 for this dataset and split.
- The learned representation-transfer model did not show the same improvement on the final test set, obtaining a test Spearman correlation of 0.200757 compared to 0.337409 for concatenation and 0.217618 for ESM-only.
- One possible explanation is that the learned mapping may not preserve all information useful for the downstream fluorescence prediction when projecting representations between two models trained under different pretraining objectives (causal autoregressive LM vs. bidirectional masked LM).
- These findings represent an empirical observation for this specific dataset, split, and implementation, and do not imply that representation transfer is generally ineffective across other architectures or protein engineering tasks.

---

## 8. Limitations

- **Small dataset:** The dataset is limited to single mutants of a single protein (avGFP).
- **Frozen pretrained representations:** Pretrained encoders remained frozen; end-to-end fine-tuning was not investigated.
- **Simple prediction heads:** Small 2-layer MLPs were used to isolate representation differences from model capacity.
- **Single source model:** Only one source model (RITA-s) was evaluated alongside ESM-2.
- **Residue-level representations:** Embeddings were extracted at the single mutated position rather than pooling whole-protein representations or structural graphs.
- **Limited transfer-method exploration:** Mapping was trained solely via MSE latent alignment without task-guided or contrastive loss formulations.

---

## 9. Reproducibility

- Random seed fixed to 42 across Python, NumPy, and PyTorch.
- Frozen pretrained encoders: `facebook/esm2_t12_35M_UR50D` and `lightonai/RITA_s`.
- Fusion weight fixed to $\alpha = 0.5$.
- Models refitted from scratch on the combined 871 development variants before generating final predictions for the 213 test variants.
- The notebook `GFP_Representation_Transfer.ipynb` contains the complete, self-contained implementation.
