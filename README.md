# Embracing Resilience: Balancing Risk Factor and Protective Factor in Dynamic Suicide Risk Prediction

This repository contains the official implementation of **DFIL** (Dynamic Factor Influence Learning), proposed in the paper *"Embracing Resilience: Balancing Risk Factor and Protective Factor in Dynamic Suicide Risk Prediction"*, which is currently under review for KDD 2026.

DFIL is a multi-task learning framework that jointly models **protective factors** (suicide resilience) and **risk factors** from users' social media post sequences, and captures their dynamic influence on subsequent suicide risk transitions via **Conditional Transfer Effect (CTE)**.

---

## Dataset Access

Dataset: **PFA (Protective Factor-Aware Dataset)** — constructed from posts collected from 6,943 users on the r/SuicideWatch subreddit (June 2010 – September 2022). Following the filtering and annotation procedure described in the paper, the final expert-annotated dataset contains 237 users and 2,515 posts, with labels for risk factors, protective factors, and temporally-aware suicide risk levels following the Columbia Suicide Severity Rating Scale (C-SSRS).

To receive access, you will need to read, sign, and send back the attached data usage agreement (DUA).

The DUA contains restrictions on how you can use the data. We would like to draw your attention to several restrictions in particular:

- You cannot transfer or reproduce any part of the data set. This includes publishing parts of users' posts.
- You cannot attempt to identify any user in the data set.
- You cannot contact any user in the data set.

If your institution has issues with language in the DUA, please have the responsible person at your institution contact us with their concerns and suggested modifications.

Once the Primary Investigator has signed the DUA, the Primary Investigator should email the signed form to **hialex.li@connect.polyu.hk**

---

## Overview

Existing suicide risk prediction methods predominantly focus on risk factors while overlooking suicide resilience provided by protective factors. Moreover, they largely adopt static paradigms that fail to capture the inherently dynamic nature of suicide risk. DFIL addresses both limitations through:

- **Post Embedding**: each post $p_t$ in a user's dynamic post sequence is encoded via Sentence-BERT (SBERT) to capture semantic and psychological state information.
- **Sequential Post Modeling**: a BiLSTM captures long-term sequential dependencies across the user's post history.
- **Temporal Attention**: a temporal decay mechanism assigns higher weights to posts that most significantly indicate evolving suicide risk patterns based on inter-post time intervals.
- **Risk and Protective Factors Learning**: two separate factor encoders (non-shared parameters) produce post-level risk factor embeddings **e⁻** and protective factor embeddings **e⁺**, supervised by multi-label auxiliary losses L_rf and L_pf.
- **Dynamic Factors Influence Learning**: factor summaries **ẽ⁺** / **ẽ⁻** are obtained via alignment-based weighted pooling (temperature τ) and used to compute the Conditional Transfer Effect (CTE) — a counterfactual masking objective that quantifies the incremental predictive benefit of each factor type on subsequent suicide risk.
- **Multi-task Decoder**: the final prediction integrates the temporal user representation **u** with both factor summaries, optimized jointly via ordinal regression loss L_sr, auxiliary factor losses L_pf / L_rf, and the CTE loss L_CTE.

**Annotation taxonomy:**

| Category | Labels |
|----------|--------|
| Suicide Risk Levels (C-SSRS) | Indicator · Ideation · Behavior · Attempt |
| Protective Factors | Coping Strategy · Psychological Capital · Sense of Responsibility · Meaning in Life · Social Support |
| Risk Factors | Suicide Means · Prior Self-Harm or Suicidal Thought/Attempt · Hopelessness · Traumatic Experience · Physical Health/Characteristic ... |

---

## Repository Structure

```
.
├── main.py                   # Entry point: training and 5-fold cross-validation
├── src/
│   ├── TempATT.py            # DFIL model (LightningModule)
│   ├── attention.py          # Temporal attention module
│   └── data_preparation.py   # PFA dataset preprocessing and SBERT embedding generation
├── utils/
│   ├── data_loader.py        # RedditDataset and collate function
│   ├── loss.py               # Ordinal regression loss, focal loss, CB loss
│   └── evaluation.py         # Graded metrics (GP, GR, FS) and classification report
└── dataset/                  # Place preprocessed .pkl files here
```

---

## Requirements

```bash
pip install torch pytorch-lightning transformers sentence-transformers scikit-learn pandas numpy
```

Tested with Python 3.8–3.11, PyTorch ≥ 1.13, PyTorch Lightning ≥ 1.9.

---

## Data Preparation

The model expects a preprocessed `.pkl` file of sliding-window post timelines with pre-computed SBERT embeddings. Each record corresponds to a window of T posts predicting the **subsequent** suicide risk level.

```bash
python src/data_preparation.py \
    --input path/to/df_final_table.xlsx \
    --window 4 \
    --output dataset/data_window4.pkl \
    --model sentence-transformers/nli-roberta-large
```

The resulting dataframe contains the following columns:

| Column | Description |
|--------|-------------|
| `author` | User identifier |
| `user_id` | Integer-mapped user ID |
| `created_utc` | Post timestamps within the observation window |
| `cur_su_y` | Historical suicide risk labels per post (Indicator/Ideation/Behavior/Attempt) |
| `cur_bp_y` | Risk factor multi-labels per post |
| `cur_bp_res` | Protective factor multi-labels per post |
| `fu_30_su_y` | Subsequent suicide risk label (prediction target) |
| `sb_1024` | SBERT post embeddings (1024-dim) |

---

## Training: 5-Fold Cross-Validation

```bash
python main.py \
    --input dataset/data_window4.pkl \
    --run_all_folds \
    --af 30 \
    --embed_type sb \
    --hidden_dim 1024 \
    --lr 5e-5 \
    --tau 0.8 \
    --lambda_te 0.2 \
    --lambda_sr 1.0 \
    --lambda_pf 0.4 \
    --lambda_rf 0.1 \
    --save checkpoints
```

---

## Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden_dim` | 1024 | BiLSTM and factor encoder hidden size |
| `--tau` | 0.8 | Temperature τ controlling sharpness of factor alignment distribution |
| `--lambda_sr` | 1.0 | Weight for ordinal regression suicide risk loss L_sr |
| `--lambda_pf` | 0.4 | Weight for protective factor auxiliary loss L_pf |
| `--lambda_rf` | 0.1 | Weight for risk factor auxiliary loss L_rf |
| `--lambda_te` | 0.2 | Weight for Conditional Transfer Effect loss L_CTE |
| `--gamma_te` | 1.0 | Scaling γ for CTE influence score sigmoid at test time |
| `--af` | 30 | Follow-up window in days; selects the subsequent risk label column |
| `--loss` | oe | Main loss type: `oe` (ordinal regression), `focal`, `ce` |
| `--val_ratio` | 0.125 | Fraction of training split used for validation |

---

## Evaluation Metrics

Following the paper's evaluation protocol:

- **GP / GR / FS**: Graded Precision, Graded Recall, and FScore — penalize misclassifications more severely when the ordinal distance between predicted and true risk level is larger.
- **WP / WR / WF1**: Weighted Precision, Recall, and F1-score across the four C-SSRS suicide risk levels.

Risk level labels: `Indicator (0)` · `Ideation (1)` · `Behavior (2)` · `Attempt (3)`.
