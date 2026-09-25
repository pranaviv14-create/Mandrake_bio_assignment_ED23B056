# Experimental Report: Representation Transfer for GFP Fluorescence Prediction

**Author:** Pranavi Voleti (ED23B056)  
**Task:** Protein fitness prediction under position-held-out splits  
**Base Model:** ESM-2 35M (`facebook/esm2_t12_35M_UR50D`)  
**Source Model:** RITA-s (`lightonai/RITA_s`, approximately 85M parameters)  
**Dataset:** avGFP Single Mutants (Sarkisyan et al., 2016 / ProteinGym v1.3)  
**Primary Metric:** Spearman Rank Correlation (rho)  
**Secondary Metric:** Mean Squared Error (MSE) in original DMS_score units  

---

## 1. Problem Statement

Predicting the fitness consequences of mutations at residue sites that have never been measured experimentally is a central challenge in protein engineering. Random train/test splits tend to overestimate model performance because variants sharing the same mutated residue share local structural context and baseline mutational tolerance.

This assignment tests models on a position-held-out split of green fluorescent protein (avGFP). We investigate whether information from an architecturally distinct protein language model family (**RITA-s**) can improve fluorescence prediction when combined with a base **ESM-2 35M** encoder.

---

## 2. Dataset and Fixed Split

The dataset contains 1,084 single-substitution variants of `GFP_AEQVI_Sarkisyan_2016` (238 amino acids each) from ProteinGym v1.3.

The split follows the official ProteinGym contiguous folds:

| Split | Variants | Folds | Mutated Positions | Permitted Use |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 657 | 1, 2, 3 | 50 to 191 (140 unique sites) | Model fitting, mapper training, scaler fitting |
| **Validation** | 214 | 4 | 192 to 237 (46 unique sites) | Hyperparameter tuning, method selection |
| **Train + Val (Refit)** | 871 | 1 to 4 | 50 to 237 (186 unique sites) | Refitting chosen pipeline from scratch |
| **Test** | 213 | 0 | 3 to 49 (47 unique sites) | Final evaluation only |

### Split Verification and Leakage Checks
- We verified programmatically that the position sets are strictly disjoint:
  `train_positions.isdisjoint(val_positions)`
  `train_positions.isdisjoint(test_positions)`
  `val_positions.isdisjoint(test_positions)`
- All sequences have length 238 and exactly one mutation.
- Scaler and target normalization statistics were fit strictly on the training set during development, and re-estimated on the combined 871 variants during the final refit. Test labels were withheld until all choices were frozen.

---

## 3. Models

1. **Base Model:** **ESM-2 35M** (`facebook/esm2_t12_35M_UR50D`)
   - 12 transformer layers, 480 hidden dimensions, ~35M parameters.
   - Pretrained with bidirectional masked language modeling on UniRef50.

2. **Source Model:** **RITA-s** (`lightonai/RITA_s`)
   - 12 causal transformer layers, 768 hidden dimensions, approximately 85M parameters.
   - Pretrained with autoregressive next-token prediction on UniRef100.
   - Represents an orthogonal modeling family (causal generative model vs. bidirectional masked LM).

Pretrained model weights remain frozen throughout all experiments.

---

## 4. Representation Choice

Because all variants in this dataset are single-residue substitutions at known 1-based biological positions $p \in [1, 238]$, we extract the hidden representation corresponding to the mutated residue:
- **ESM-2 35M:** Tokenizer prepends `<cls>` at index 0, so biological position $p$ maps directly to token index $p$. Feature dimension: 480.
- **RITA-s:** No start token is prepended, so biological position $p$ maps to token index $p - 1$. Feature dimension: 768.

Extracted features were cached on disk as `.npy` arrays for fast and reproducible experimentation.

---

## 5. Baselines

To separate transfer effects from predictor capacity, all models use an identical prediction head architecture:
`Linear(in_dim, 128) -> ReLU() -> Dropout(0.1) -> Linear(128, 1)`

We evaluate three baselines:
1. **ESM-only:** Input dimension 480.
2. **RITA-only:** Input dimension 768.
3. **Simple Concatenation:** Concatenated `[ESM ; RITA]` features (dimension $480 + 768 = 1248$).

---

## 6. Representation Transfer Method

Inspired by SoupFold (Jang et al., 2026), we adapted the representation-mapping idea to residue sequence embeddings:
1. **Transfer Network:** A 2-layer MLP (`Linear(768, 512) -> ReLU() -> Linear(512, 480)`).
2. **Alignment Loss:** Trained strictly on the 657 training variants using Mean Squared Error to map RITA representations into ESM latent space:
   $$\mathcal{L}_{\text{map}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \| f(z_{\text{RITA}, i}) - z_{\text{ESM}, i} \|_2^2$$
3. **Latent Space Fusion:** The mapped RITA representation is combined with the ESM representation:
   $$H = \frac{z_{\text{ESM}} + \alpha \cdot f(z_{\text{RITA}})}{1 + \alpha} \quad (\alpha = 0.5)$$
   We chose $\alpha = 0.5$ so the base ESM representation is weighted twice as heavily as the transferred representation, anchoring predictions primarily to the base model while incorporating source signals.
4. **Fluorescence Regressor:** An MLP prediction head trained on $H$ to predict `DMS_score`.

---

## 7. Development and Validation Procedure

Models were trained on the 657 training variants with AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 32) using early stopping on the 214 validation variants (patience = 10 epochs).

### Validation Results (Positions 192 to 237)
| Method | Validation Spearman (rho) | Validation MSE | Selected Epochs |
| :--- | :---: | :---: | :---: |
| **ESM-only** | 0.2027 | 0.3908 | 20 |
| **RITA-only** | 0.1651 | 0.3979 | 20 |
| **ESM + RITA (Concatenation)** | **0.2408** | **0.3069** | 20 |
| **Representation Transfer** | 0.1720 | 0.3537 | 20 |

---

## 8. Final Refit Procedure

Once model architectures and hyperparameters were frozen:
1. All learned components (mappers, prediction heads) and feature scalers were reinitialized.
2. Models were refit from scratch using all **871 development variants** (Train + Validation).
3. The refitted models generated predictions for the 213 test variants (positions 3 to 49).
4. Prediction files were saved in submission format (`variant_id,predicted_fitness`).

---

## 9. Final Test Results

After predictions were frozen, metrics were computed against the ground-truth test labels:

| Method | Test Spearman (rho) | Test MSE | Relative vs. ESM |
| :--- | :---: | :---: | :---: |
| **ESM-only** | 0.2175 | 0.4636 | Baseline |
| **RITA-only** | 0.1817 | 0.4293 | -16.5% |
| **ESM + RITA (Concatenation)** | **0.2764** | **0.4256** | **+27.1%** |
| **Representation Transfer** | 0.2006 | 0.4405 | -7.8% |

---

## 10. Discussion

1. **Does RITA provide complementary information?**  
   In this experiment, the improvement of the concatenation model over the ESM-only baseline (test rho = 0.2764 vs. 0.2175) suggests that the RITA representation contains information that can be useful alongside ESM. Despite RITA having lower standalone accuracy (rho = 0.1817), combining both representations improved ranking performance.

2. **Did learned representation transfer help?**  
   The particular transfer method did not outperform concatenation; therefore representation transfer was not clearly beneficial in this experiment (test rho = 0.2006 vs. 0.2764 for concatenation and 0.2175 for ESM-only).
   
   A plausible explanation is that ESM-2 and RITA-s were trained with very different objectives (bidirectional MLM vs. left-to-right causal LM). Forcing RITA representations into ESM latent space through an unconstrained MSE mapper trained on 657 examples compressed the representation, washing out the unique signals that RITA contributed when simply concatenated.

3. **Comparison with SoupFold:**  
   SoupFold operated on pairwise representations with spatial diffusion supervision for complex structure prediction. For single-residue fitness regression, unconstrained concatenation preserved the separate representation channels better than linear latent averaging.

---

## 11. Limitations

1. **Scope:** These results apply to single-substitution variants of avGFP across the tested position split. They do not demonstrate that concatenation or transfer will behave identically across other protein families or multi-mutant fitness landscapes.
2. **Alignment Objective:** We used simple MSE regression for representation transfer without task-specific fitness supervision during the mapping step. Alternative alignment losses (e.g. contrastive or task-guided losses) might preserve fitness-relevant dimensions differently.

---

## 12. Conclusion

- Combining representations from an autoregressive model (RITA-s) and a masked model (ESM-2 35M) via concatenation improved fluorescence prediction at held-out mutation positions over ESM-2 alone in this dataset.
- Direct latent space mapping via MSE did not outperform simple concatenation and produced a negative transfer effect relative to the concatenation baseline.
- This negative result is reported factually, following the assignment guidelines that improvement is an empirical research question.
