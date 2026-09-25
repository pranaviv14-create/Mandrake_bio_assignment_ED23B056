# GFP Representation Transfer

**Author:** Pranavi Voleti (ED23B056)  
**Task:** Protein fitness prediction under position-held-out splits  
**Base Model:** ESM-2 35M (`facebook/esm2_t12_35M_UR50D`)  
**Source Model:** RITA-s (`lightonai/RITA_s`, approximately 85M parameters)  
**Dataset:** avGFP Single Mutants (Sarkisyan et al., 2016 / ProteinGym v1.3)  
**Primary Metric:** Spearman Rank Correlation ($\rho$)  
**Secondary Metric:** Mean Squared Error (MSE) in original `DMS_score` units  

---

## 1. Problem Statement

Predicting the fitness consequences of mutations at residue positions never seen during training is a critical challenge in protein engineering. Standard random train/test splits overestimate model generalization because variants with mutations at the same position share local backbone geometry and baseline mutational tolerance.

This project investigates whether representations from an architecturally distinct protein language model (**RITA-s**) can improve green fluorescent protein (avGFP) fluorescence prediction when combined with a base **ESM-2 35M** encoder under a strict **position-held-out split**.

---

## 2. Dataset and Contiguous Fold Split

The dataset consists of 1,084 single-substitution variants of `GFP_AEQVI_Sarkisyan_2016` (238 amino acids each) from ProteinGym v1.3:

| Split | Variants | Folds | Mutated Positions | Permitted Use |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 657 | 1, 2, 3 | 50 to 191 (140 unique sites) | Model fitting, mapper training, scaler fitting |
| **Validation** | 214 | 4 | 192 to 237 (46 unique sites) | Hyperparameter tuning, method selection |
| **Train + Val (Refit)** | 871 | 1 to 4 | 50 to 237 (186 unique sites) | Refitting chosen pipeline from scratch |
| **Test** | 213 | 0 | 3 to 49 (47 unique sites) | Final evaluation only |

### Split Verification & Zero-Leakage Guarantee
- The mutated position sets are strictly disjoint between Train, Validation, and Test sets:
  - $\text{Train Positions} \cap \text{Val Positions} = \emptyset$
  - $\text{Train Positions} \cap \text{Test Positions} = \emptyset$
  - $\text{Val Positions} \cap \text{Test Positions} = \emptyset$
- All sequences have length 238 and contain exactly one substituted residue.
- Feature scalers and target statistics were fit strictly on training data during development, and re-estimated on the combined 871 variants during the final refit. Test labels were withheld until all model decisions were frozen.

---

## 3. Models and Representation Extraction

1. **Base Model:** **ESM-2 35M** (`facebook/esm2_t12_35M_UR50D`)
   - 12 transformer layers, 480 hidden dimensions, ~35M parameters.
   - Bidirectional masked language model pretrained on UniRef50.
   - Prepends a `<cls>` token at index 0; biological 1-based position $p$ maps to token index $p$.

2. **Source Model:** **RITA-s** (`lightonai/RITA_s`)
   - 12 causal transformer layers, 768 hidden dimensions, ~85M parameters.
   - Autoregressive causal language model pretrained on UniRef100.
   - No prepended start token; biological 1-based position $p$ maps to token index $p - 1$.

Both pretrained encoders remain **strictly frozen** (`requires_grad = False`, `.eval()`). Features are extracted at the specific mutated residue position.

### RITA Tokenizer Verification
The canonical vocabulary was verified against `AutoTokenizer.from_pretrained("lightonai/RITA_s")`. Every standard amino acid token and special token matches the official vocabulary table exactly (IDs 3 to 22 for canonical amino acids, `<unk>`: 0, `<PAD>`: 1, `<EOS>`: 2). The canonical mapping was kept because HuggingFace's `tokenizer.json` for RITA-s contains an upstream normalizer bug that replaces $Z \to E$ and outputs ID 26 for Glutamate, exceeding the model's vocabulary size of 26 ($0 \dots 25$). The canonical vocabulary maps $E \to 7$, ensuring $\max(\text{input\_ids}) < 26$ without requiring any `torch.clamp` workaround.

---

## 4. Methods Compared

To isolate transfer effects from predictor capacity, all methods employ an identical regressor architecture:
$$\text{Linear}(\text{in\_dim}, 128) \to \text{ReLU}() \to \text{Dropout}(0.1) \to \text{Linear}(128, 1)$$

1. **ESM-only baseline:** Frozen ESM-2 residue embedding (480-dim) $\to$ MLP regressor.
2. **RITA-only baseline:** Frozen RITA-s residue embedding (768-dim) $\to$ MLP regressor.
3. **Simple Concatenation:** Concatenated $[\mathbf{z}_{\text{ESM}} \,;\, \mathbf{z}_{\text{RITA}}]$ features (1248-dim) $\to$ MLP regressor.
4. **Representation Transfer (SoupFold-Inspired):**
   - **Mapper:** 2-layer MLP ($\text{Linear}(768, 512) \to \text{ReLU}() \to \text{Linear}(512, 480)$) trained with MSE loss on training variants to map RITA representations into ESM latent space.
   - **Latent Fusion:** Combined via weighted average:
     $$H = \frac{\mathbf{z}_{\text{ESM}} + \alpha \cdot f(\mathbf{z}_{\text{RITA}})}{1 + \alpha} \quad (\alpha = 0.5)$$
     Setting $\alpha = 0.5$ weights the base ESM representation twice as heavily as the transferred RITA representation.
   - **Fitness Regressor:** MLP regressor trained on $H$ to predict `DMS_score`.

---

## 5. Training and Refit Protocol

- **Optimizer:** AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 32).
- **Epochs:** 20 epochs was used as the minimum training duration with early stopping (patience 10).
- **Final Refit:** Once hyperparameters were fixed, all learned mappers and regressors were reinitialized and refitted from scratch on all **871 development variants** (Train + Validation) before evaluating on the 213 test variants.

---

## 6. Results

### Validation Performance (Fold 4, Positions 192 to 237, N = 214)

| Method | Validation Spearman ($\rho$) | Validation MSE |
| :--- | :---: | :---: |
| **ESM-only** | 0.2027 | 0.3908 |
| **RITA-only** | 0.1651 | 0.3979 |
| **ESM + RITA (Concatenation)** | **0.2408** | **0.3069** |
| **Representation Transfer** | 0.1720 | 0.3537 |

### Final Test Performance (Fold 0, Positions 3 to 49, N = 213)

| Method | Test Spearman ($\rho$) | Test MSE | Relative vs. ESM-only |
| :--- | :---: | :---: | :---: |
| **ESM-only** | 0.2175 | 0.4636 | Baseline |
| **RITA-only** | 0.1817 | 0.4293 | -16.5% |
| **ESM + RITA (Concatenation)** | **0.2764** | **0.4256** | **+27.1%** |
| **Representation Transfer** | 0.2006 | 0.4405 | -7.8% |

---

## 7. Discussion and Scientific Observations

1. **Does RITA-s provide complementary information?**  
   Yes. Simple concatenation of ESM-2 and RITA-s representations achieved the highest Spearman correlation ($\rho = 0.2764$, a +27.1% relative gain over ESM-only). This indicates that the autoregressive causal model encodes sequence patterns that complement the bidirectional masked model.

2. **Did learned representation transfer help?**  
   No. Transferring RITA representations into ESM latent space via an unconstrained MSE mapper yielded a test correlation of $\rho = 0.2006$, underperforming concatenation ($\rho = 0.2764$) and slightly trailing ESM-only ($\rho = 0.2175$).  
   One possible explanation is that projecting the RITA representation into the ESM representation space may discard information that is retained by direct concatenation. In addition, the two models use fundamentally different pretraining objectives (autoregressive causal LM vs. bidirectional MLM), and an unconstrained MSE mapper trained on 657 examples may not capture the complex correspondence between the two latent spaces without task-specific supervision.

3. **Assignment Requirement:**  
   The assignment guidelines emphasize that improvement is an empirical research question and a well-supported negative result is acceptable. This experiment provides clear, unforced evidence distinguishing simple feature combination from representation transfer.

---

## 8. Repository Structure

```
Mandrake_bio_assignment_ED23B056/
├── .gitignore
├── GFP_Representation_Transfer.ipynb    # Clean, end-to-end runnable notebook
├── README.md                            # Complete assignment report and documentation
├── requirements.txt                     # Minimal Python dependencies
│
├── dataset/
│   ├── train.csv                        # 657 train variants (Folds 1, 2, 3)
│   ├── validation.csv                   # 214 validation variants (Fold 4)
│   ├── test.csv                         # 213 test variants (Fold 0)
│   ├── reference.fasta                  # Wildtype avGFP sequence
│   └── submission_template.csv          # Submission format template
│
├── embeddings/                          # Cached numpy embeddings (~5 MB)
│   ├── esm_train.npy
│   ├── esm_validation.npy
│   ├── esm_test.npy
│   ├── source_train.npy
│   ├── source_validation.npy
│   └── source_test.npy
│
└── results/
    ├── submission_esm.csv               # ESM-only test predictions
    ├── submission_rita.csv              # RITA-only test predictions
    ├── submission_concat.csv            # Concatenation test predictions
    ├── submission_transfer.csv          # Representation transfer test predictions
    └── figures/
        ├── dms_distribution.png         # DMS score distribution across splits
        ├── validation_vs_test_spearman.png # Validation vs Test Spearman comparison
        └── test_predictions_scatter.png # Predicted vs actual test scatter plots
```

---

## 9. How to Run

### Option A: Google Colab
1. Open Google Colab and upload or open `GFP_Representation_Transfer.ipynb`.
2. Select a GPU runtime: **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU** (CPU also supported).
3. If running directly without cloning beforehand, the first cell will automatically clone the repository from GitHub:
   ```python
   # Automatically handled in Cell 1 if dataset/ is missing:
   !git clone https://github.com/pranaviv14-create/Mandrake_bio_assignment_ED23B056.git
   %cd Mandrake_bio_assignment_ED23B056
   ```
4. Run all cells sequentially. Pre-extracted embeddings are loaded from `embeddings/` by default, allowing all models, training, refit, and evaluation to complete in under 1 minute. Setting `RECOMPUTE_EMBEDDINGS = True` will re-extract embeddings from the raw transformer weights.

### Option B: Local Setup
```bash
git clone https://github.com/pranaviv14-create/Mandrake_bio_assignment_ED23B056.git
cd Mandrake_bio_assignment_ED23B056
pip install -r requirements.txt
jupyter notebook GFP_Representation_Transfer.ipynb
```
