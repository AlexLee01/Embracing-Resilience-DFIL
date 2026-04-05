# Embracing Resilience: Dual-Factor Informed Learning for Suicide Risk Assessment

This repository contains the implementation of **DFIL** (Dual-Factor Informed Learning), a framework for longitudinal suicide risk assessment from social media posts. DFIL jointly models **protective factors** (resilience) and **risk factors** alongside temporal post dynamics, using a Transfer-Effect-inspired auxiliary loss to explicitly capture how each factor type shifts the model's predictive confidence.

---

## Dataset Access

Dataset of the paper: *"Embracing Resilience: Balancing Risk Factor and Protective Factor in Dynamic Suicide Risk Prediction"*

To receive access, you will need to read, sign, and send back the attached data usage agreement (DUA).

The DUA contains restrictions on how you can use the data. We would like to draw your attention to several restrictions in particular:

- You cannot transfer or reproduce any part of the data set. This includes publishing parts of users' posts.
- You cannot attempt to identify any user in the data set.
- You cannot contact any user in the data set.

If your institution has issues with language in the DUA, please have the responsible person at your institution contact us with their concerns and suggested modifications.

Once the Primary Investigator has signed the DUA, the Primary Investigator should email the signed form to **hialex.li@connect.polyu.hk**

---

## Overview

Suicide risk assessment on social media faces two challenges:
1. **Longitudinal modeling**: user mental states evolve over time across multiple posts.
2. **Factor imbalance**: protective factors (e.g., coping strategy, sense of responsibility) are systematically under-modeled relative to risk factors.

DFIL addresses both through:
- A **bidirectional LSTM** with **time-sensitive attention** to encode post sequences weighted by inter-post time gaps.
- **Dual factor encoders** that separately learn protective-factor and risk-factor representations via auxiliary multi-label classification.
- **Softmax-weighted factor aggregation** (temperature τ) to produce user-level protective and risk embeddings.
- **Four prediction heads** conditioned on: baseline, +protective, +risk, and +both factor embeddings.
- A **Transfer Effect (TE) loss** that rewards factor representations whose inclusion increases the model's probability assigned to the true class.

---

## Repository Structure

```
.
├── main.py                   # Entry point: training and 5-fold evaluation
├── src/
│   ├── TempATT.py            # Core model (LightningModule)
│   ├── attention.py          # Learnable attention module
│   └── data_preparation.py   # Dataset preprocessing and embedding generation
├── utils/
│   ├── data_loader.py        # Dataset class and collate function
│   ├── loss.py               # Loss functions (ordinal entropy, focal, CB)
│   └── evaluation.py         # Metrics (GP, GR, FS, classification report)
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

The model expects a preprocessed `.pkl` file containing per-user sliding-window timelines with pre-computed sentence embeddings.

To prepare data from the raw Excel annotation file:

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
| `created_utc` | List of post timestamps (window) |
| `cur_su_y` | Historical suicide risk labels per post |
| `cur_bp_y` | Protective factor labels per post (multi-label) |
| `cur_bp_res` | Risk factor labels per post (multi-label) |
| `fu_30_su_y` | Future suicide risk label (prediction target) |
| `sb_1024` | List of sentence embeddings (1024-dim) |

---

## Training

### Single fold

```bash
python main.py \
    --input dataset/data_window4.pkl \
    --n_fold 1 \
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

### 5-fold cross-validation

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
| `--hidden_dim` | 1024 | LSTM and encoder hidden size |
| `--tau` | 0.8 | Temperature for factor aggregation softmax |
| `--lambda_sr` | 1.0 | Weight for main suicide risk loss |
| `--lambda_pf` | 0.4 | Weight for protective factor auxiliary loss |
| `--lambda_rf` | 0.1 | Weight for risk factor auxiliary loss |
| `--lambda_te` | 0.2 | Weight for Transfer Effect loss |
| `--gamma_te` | 1.0 | Scaling for TE sigmoid at test time |
| `--af` | 30 | Follow-up window (days); determines target label column |
| `--loss` | oe | Main loss type: `oe` (ordinal entropy), `focal`, `ce` |
| `--val_ratio` | 0.125 | Fraction of training set used for validation |

---

## Evaluation Metrics

- **GP / GR / FS**: Graded Precision, Recall, and F-score penalizing larger ordinal errors more severely.
- **Weighted P / R / F1**: Standard weighted-average classification metrics across 4 risk classes.
- **Per-class report**: Printed via `sklearn.metrics.classification_report` for all three tasks (suicide risk, protective factors, risk factors).

Risk classes: `su_indicator (0)`, `su_ideation (1)`, `su_behavior (2)`, `su_attempt (3)`.

---

## Model Architecture

```
Post embeddings (1024-dim)
        │
   ┌────▼─────────────────────────────────────┐
   │  Protective Encoder   Risk Encoder        │  (auxiliary decoders → multi-label loss)
   └────┬──────────────────────┬──────────────┘
        │                      │
   Bi-LSTM + Time-Sensitive Attention
        │
   User repr h
        │
   ┌────▼────────────────────────────────────┐
   │  Factor Aggregation (cosine sim, τ)     │
   │  e_plus (protective)  e_minus (risk)    │
   └────┬──────────────────────┬────────────┘
        │                      │
   ┌────▼──────────────────────▼────────────┐
   │  4 Heads: base / +pf / +rf / +both     │
   └────────────────────────────────────────┘
        │
   Ordinal Entropy Loss + TE Loss
```

---

## Citation

If you use this code, please cite our paper (citation to be updated upon publication).
