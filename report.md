# Experimental Report: Representation Transfer for GFP Fluorescence Prediction

**Author:** Student Researcher  
**Assignment:** Representation Transfer for Protein Fitness Prediction  
**Base Model:** ESM-2 35M (`facebook/esm2_t12_35M_UR50D`)  
**Source Model:** RITA-s (`lightonai/RITA_s`)  
**Dataset:** Sarkisyan et al. (2016) / ProteinGym v1.3 avGFP Single-Substitution Variants  
**Primary Metric:** Spearman Rank Correlation ($\rho$) on 213 held-out test variants  
**Secondary Metric:** Mean Squared Error (MSE) in original `DMS_score` units  

---

## 1. Problem Statement

Predicting the fitness consequences of mutations at residue sites that have never been experimentally assayed is a core challenge in protein engineering. When evaluated on random train/test splits, machine-learning models frequently rely on positional memorization—leveraging known measurements at a given residue to predict other mutations at that same site. 

To evaluate genuine generalization across protein positions, this assignment tests models on a **position-held-out split** within green fluorescent protein (avGFP). Pretrained protein language models encode different inductive biases depending on their architectural families (e.g., bidirectional masked language models vs. autoregressive causal generative models). This study investigates whether transferring representations from a distinct model family (**RITA-s**) into a base **ESM-2 35M** encoder improves fluorescence prediction on held-out mutated positions.

---

## 2. Dataset and Fixed Split

The dataset comprises all 1,084 single-substitution variants of `GFP_AEQVI_Sarkisyan_2016` (238 amino acids each) from ProteinGym v1.3. Each sequence contains exactly one substitution relative to the wildtype reference.

The experiments adhere strictly to the instructor-provided contiguous fold split:

| Split | Variants | Folds | Mutated Positions | Permitted Use |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 657 | 1, 2, 3 | 50–191 (140 unique sites) | Model fitting, representation mapper training, scaler fitting |
| **Validation** | 214 | 4 | 192–237 (46 unique sites) | Hyperparameter tuning, method selection, early stopping |
| **Train + Val (Refit)** | 871 | 1–4 | 50–237 (186 unique sites) | Re-initialization & full refit before final test evaluation |
| **Test** | 213 | 0 | 3–49 (47 unique sites) | Final evaluation only. Ground truth labels withheld during development |

### Data Leakage Controls
- Zero positional overlap: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- Zero sequence overlap: Sequences in different splits differ from the reference at distinct positions.
- Feature normalization and target standardization statistics were computed strictly on the training partition during model selection, and re-estimated on the combined 871 training+validation variants for the final refit.

---

## 3. Pretrained Models & Representation Choice

1. **Base Model — ESM-2 35M (`facebook/esm2_t12_35M_UR50D`):**
   - 12 transformer layers, 480 hidden dimensions, 35M parameters.
   - Trained with a bidirectional masked language modeling objective across UniRef50.
2. **Source Model — RITA-s (`lightonai/RITA_s`):**
   - 12 causal transformer layers, 768 hidden dimensions, 24M parameters.
   - Trained as an autoregressive generative language model (next-token prediction) across UniRef100.
   - *Rationale:* RITA represents an orthogonal modeling paradigm (causal autoregressive generation) compared to ESM-2 (bidirectional masked reconstruction) while remaining computationally lightweight and Colab-compatible.

### Token Alignment and Residue-Level Extraction
Because all variants are single-amino-acid substitutions at known 1-based biological positions $p \in [1, 238]$, we extract the local hidden representation corresponding to the mutated residue rather than a sequence-wide mean pool:
- **ESM-2 35M:** Tokenizer prepends `<cls>` at index 0. Therefore, biological position $p$ maps directly to token index $p$ in the output tensor. Feature vector: $z_{\text{ESM}} \in \mathbb{R}^{480}$.
- **RITA-s:** No prepended start token. Biological position $p$ maps to token index $p - 1$. Feature vector: $z_{\text{RITA}} \in \mathbb{R}^{768}$.

Embeddings were precomputed and cached on disk (`.npy`) to ensure reproducibility and computational efficiency.

---

## 4. Baselines

To isolate representation transfer from general predictor capacity, three baselines were implemented using an identical prediction head architecture:
$$\text{Linear}(d_{\text{in}}, 128) \to \text{ReLU}() \to \text{Dropout}(0.1) \to \text{Linear}(128, 1)$$

1. **Baseline 1 (ESM-only):** Input $z_{\text{ESM}} \in \mathbb{R}^{480} \to$ Predictor $\to \widehat{\text{DMS}}$.
2. **Baseline 2 (Source-only RITA):** Input $z_{\text{RITA}} \in \mathbb{R}^{768} \to$ Predictor $\to \widehat{\text{DMS}}$.
3. **Baseline 3 (Simple Concatenation):** Input $[z_{\text{ESM}} \,\|\, z_{\text{RITA}}] \in \mathbb{R}^{1248} \to$ Predictor $\to \widehat{\text{DMS}}$.  
   *Note:* Concatenation evaluates whether the combined features contain complementary information, providing a critical benchmark against which learned representation transfer is compared.

---

## 5. Representation Transfer Method (SoupFold-Inspired)

Inspired by the representation alignment mechanism in **SoupFold** (Jang et al., 2026), we adapted the core idea from 2D residue pairs to 1D residue embeddings:
1. **Transfer Network ($f_{\text{RITA} \to \text{ESM}}$):** A 2-layer MLP ($\text{Linear}(768, 512) \to \text{ReLU}() \to \text{Linear}(512, 480)$).
2. **Representation Alignment Loss:** Trained exclusively on the 657 development training variants using Mean Squared Error between the mapped RITA embedding and the target ESM embedding:
   $$\mathcal{L}_{\text{map}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \| f_{\text{RITA} \to \text{ESM}}(z_{\text{RITA}, i}) - z_{\text{ESM}, i} \|_2^2$$
3. **Latent Space Fusion:** The mapped source representation is blended with the base ESM representation:
   $$H_i = \frac{z_{\text{ESM}, i} + \alpha \cdot f_{\text{RITA} \to \text{ESM}}(z_{\text{RITA}, i})}{1 + \alpha} \quad (\alpha = 0.5)$$
4. **Fluorescence Regression:** Downstream prediction head trained on $H_i \in \mathbb{R}^{480}$ to predict continuous `DMS_score`.

---

## 6. Experimental Results

### 6.1 Validation Set Performance (Folds 1–3 Train $\to$ Fold 4 Val, Positions 192–237)

Models were fit on the 657 training variants and evaluated on the 214 validation variants with early stopping (patience = 10 epochs):

| Method | Validation Spearman $\rho$ | Validation MSE | Selected Epochs |
| :--- | :---: | :---: | :---: |
| **Baseline 1: ESM-only** | 0.2027 | 0.3908 | 20 |
| **Baseline 2: Source-only (RITA-s)** | 0.1651 | 0.3979 | 20 |
| **Baseline 3: Concatenation** | **0.2408** | **0.3069** | 20 |
| **Method 4: Representation Transfer** | 0.1720 | 0.3537 | 20 |

### 6.2 Final Model Refit & Test Set Performance (Folds 1–4 Refit $\to$ Fold 0 Test, Positions 3–49)

All models and feature scalers were reinitialized and refitted from scratch on the combined 871 development variants ($\text{Train} + \text{Val}$). Test predictions were then generated for the 213 test variants and evaluated against ground-truth labels:

| Method | Test Spearman $\rho$ | Test MSE | Relative $\Delta\rho$ vs. ESM |
| :--- | :---: | :---: | :---: |
| **Baseline 1: ESM-only** | 0.2175 | 0.4636 | — |
| **Baseline 2: Source-only (RITA-s)** | 0.1817 | 0.4293 | -16.5% |
| **Baseline 3: Concatenation** | **0.2764** | **0.4256** | **+27.1%** |
| **Method 4: Representation Transfer** | 0.2006 | 0.4405 | -7.8% |

---

## 7. Analysis & Discussion

### 1. Does the second model contain complementary information?
**Yes.** The feature concatenation baseline outperformed the ESM-only baseline on both validation ($\rho = 0.2408$ vs. $0.2027$) and test ($\rho = 0.2764$ vs. $0.2175$). This demonstrates that despite RITA-s having lower standalone performance than ESM-2 ($\rho = 0.1817$ test), its autoregressive generative representations provide complementary signals that enhance positional generalization when retained as distinct features. This aligns with the insight from SoupFold that standalone accuracy does not dictate representation utility.

### 2. Does explicit representation mapping provide an advantage over concatenation?
**No.** Directly projecting RITA embeddings into ESM-2 latent space via an unconstrained MSE regression mapper resulted in a test correlation of $\rho = 0.2006$, which is lower than both simple concatenation ($\rho = 0.2764$) and standalone ESM-2 ($\rho = 0.2175$).

### 3. Mechanistic Explanation:
- **Geometry of Latent Spaces:** ESM-2 and RITA-s were trained under fundamentally different objectives (bidirectional MLM vs. left-to-right causal LM). Forcing RITA embeddings into ESM space via an MSE mapper with 657 examples causes an information bottleneck, projecting the source representation into the dominant variance modes of ESM and washing out the unique directional signals that made concatenation effective.
- **Context of SoupFold:** SoupFold operated on residue-pair representations with strong geometric structure in the context of co-folding diffusion trunks. For 1D sequence-level residue fitness regression, unconstrained feature concatenation allows the MLP regressor to learn linear combinations directly rather than forcing intermediate subspace compression.

---

## 8. Limitations & Remaining Uncertainties

1. **Target-Guided Alignment:** The transfer mapper in this study was trained purely with representation-space MSE without target fluorescence guidance. Jointly training the mapper with task loss or using contrastive alignment may better preserve fitness-relevant dimensions.
2. **Single-Site vs. Epistatic Transfer:** All variants in this study were single-point substitutions. The transfer characteristics between autoregressive and masked models may differ markedly on higher-order multi-mutant fitness landscapes.
3. **Dataset Scope:** The findings are established on avGFP. Although position-held-out splits rigorously test intra-protein generalization, cross-protein family transfer remains to be explored.

---

## 9. Conclusion

- Representations from a different protein model family (autoregressive RITA-s) provide complementary evolutionary signal that substantially improves out-of-position GFP fluorescence prediction over ESM-2 35M alone when combined via concatenation ($\rho = 0.2764$ vs. $0.2175$).
- Naive subspace mapping into the base model space via MSE acts as a lossy projection, confirming that preserving separate representation channels is preferable to unguided latent averaging for this task.
- All code, cached embeddings, prediction files, and models are fully reproducible under the provided repository structure.
