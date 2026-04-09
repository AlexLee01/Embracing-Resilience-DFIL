import os
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule
from transformers import AdamW

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from src.feature_selection import select_factors_rf, apply_factor_selection
from utils.loss import loss_function
from utils.data_loader import RedditDataset, pad_collate_reddit
from utils.evaluation import *
from src.attention import Attention


def split_folds(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    n_splits: int,
    fold_idx: int,
    seed: int,
    val_ratio: float,
):
    labels = df[label_col].to_numpy()
    groups = df[group_col].to_numpy() if group_col in df.columns else None

    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(labels)), labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(labels)), labels, groups)

    train_idx, test_idx = None, None
    for i, (train_i, test_i) in enumerate(split_iter):
        if i == fold_idx:
            train_idx, test_idx = train_i, test_i
            break
    if train_idx is None:
        raise ValueError(f"fold_idx {fold_idx} out of range for n_splits={n_splits}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    if val_ratio > 0:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=val_ratio,
            random_state=seed,
            stratify=train_df[label_col].to_numpy(),
        )
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        train_df = train_df.iloc[train_idx].reset_index(drop=True)
    else:
        val_df = train_df.iloc[:0].reset_index(drop=True)

    return train_df, val_df, test_df


class TempATT(LightningModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config

        self.embed_type = self.config['embed_type'] + "_" + str(self.config['hidden_dim'])
        self.embed_layer = nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'])
        self.lstm = nn.LSTM(
            input_size=self.config['hidden_dim'],
            hidden_size=int(self.config['hidden_dim'] / 2),
            num_layers=2,
            bidirectional=True
        )

        self.time_var = nn.Parameter(torch.randn((2)), requires_grad=True)
        self.atten = Attention(hidden_size=self.config['hidden_dim'], batch_first=True)
        self.dropout = nn.Dropout(self.config['dropout'])

        # Suicide risk prediction heads
        self.suicide_base = nn.Sequential(
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['s_y_num'])
        )
        self.suicide_protective = nn.Sequential(
            nn.Linear(self.config['hidden_dim'] * 2, self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['s_y_num'])
        )
        self.suicide_risk = nn.Sequential(
            nn.Linear(self.config['hidden_dim'] * 2, self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['s_y_num'])
        )
        self.suicide_both = nn.Sequential(
            nn.Linear(self.config['hidden_dim'] * 3, self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['s_y_num'])
        )

        # Factor encoders (protective and risk)
        self.b_factor_encoder = nn.Sequential(
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU()
        )
        self.b_decoder = nn.Linear(self.config['hidden_dim'], self.config['b_y_num'])

        self.res_factor_encoder = nn.Sequential(
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU()
        )
        self.res_decoder = nn.Linear(self.config['hidden_dim'], self.config['res_y_num'])

        # Loss weights
        self.lambda_sr = float(self.config.get('lambda_sr', 1.0))
        self.lambda_pf = float(self.config.get('lambda_pf', 1.0))
        self.lambda_rf = float(self.config.get('lambda_rf', 1.0))
        self.tau = float(self.config.get('tau', 0.6))
        self.lambda_te = float(self.config.get('lambda_te', 0.1))
        self.gamma_te = float(self.config.get('gamma_te', 1.0))

    def forward(self, cur_su_y, s_y, b_y, res_y, p_num, tweets, time_interval, raw_timestamps, mode='train'):
        x = self.dropout(tweets)

        # Protective factor encoder (auxiliary task)
        b_encoded = self.b_factor_encoder(x)
        b_out = self.b_decoder(b_encoded)
        logits_b = nn.utils.rnn.pack_padded_sequence(b_out, p_num.cpu(), batch_first=True, enforce_sorted=False)[0]
        b_y = nn.utils.rnn.pack_padded_sequence(b_y, p_num.cpu(), batch_first=True, enforce_sorted=False)[0]
        b_loss = nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')(logits_b, b_y)

        # Risk factor encoder (auxiliary task)
        res_encoded = self.res_factor_encoder(x)
        res_out = self.res_decoder(res_encoded)
        logits_res = nn.utils.rnn.pack_padded_sequence(res_out, p_num.cpu(), batch_first=True, enforce_sorted=False)[0]
        res_y = nn.utils.rnn.pack_padded_sequence(res_y, p_num.cpu(), batch_first=True, enforce_sorted=False)[0]
        res_loss = nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')(logits_res, res_y)

        # Temporal LSTM encoding
        x = nn.utils.rnn.pack_padded_sequence(x, p_num.cpu(), batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Time-sensitive attention
        time_interval_for_attention = torch.exp(self.time_var[0]) * time_interval + self.time_var[0]
        time_interval_for_attention = torch.sigmoid(time_interval_for_attention + self.time_var[1])
        x = x + x * time_interval_for_attention.unsqueeze(-1)
        h, att_score = self.atten(x, p_num.cpu())

        if h.dim() == 1:
            h = h.unsqueeze(0)

        # Aggregate protective and risk factor representations
        e_plus, alpha_plus, sim_plus = self.aggregate_factors(h, b_encoded, p_num, self.tau)
        e_minus, alpha_minus, sim_minus = self.aggregate_factors(h, res_encoded, p_num, self.tau)

        # Four prediction heads
        logits_base = self.suicide_base(self.dropout(h))
        logits_plus = self.suicide_protective(self.dropout(torch.cat([h, e_plus], dim=-1)))
        logits_minus = self.suicide_risk(self.dropout(torch.cat([h, e_minus], dim=-1)))
        logits_both = self.suicide_both(self.dropout(torch.cat([h, e_plus, e_minus], dim=-1)))

        # Main task loss
        s_loss_raw = loss_function(logits_both, s_y, self.config['loss'], self.config['s_y_num'], 1.8)
        s_loss = self.lambda_sr * s_loss_raw
        b_loss = self.lambda_pf * b_loss
        res_loss = self.lambda_rf * res_loss

        # Transfer Effect (TE) loss: information gain from factor-conditioned predictions
        eps = 1e-8
        pi_base = F.softmax(logits_base, dim=-1)
        pi_plus = F.softmax(logits_plus, dim=-1)
        pi_minus = F.softmax(logits_minus, dim=-1)
        pi_both = F.softmax(logits_both, dim=-1)

        s_y_idx = s_y.unsqueeze(1)
        p_base = pi_base.gather(1, s_y_idx).squeeze(1)
        p_plus = pi_plus.gather(1, s_y_idx).squeeze(1)
        p_minus = pi_minus.gather(1, s_y_idx).squeeze(1)
        p_both = pi_both.gather(1, s_y_idx).squeeze(1)

        te_p = torch.log(p_plus + eps) - torch.log(p_base + eps)
        te_r = torch.log(p_minus + eps) - torch.log(p_base + eps)
        te_loss = self.lambda_te * (-(te_p + te_r).mean())

        total_loss = s_loss + b_loss + res_loss + te_loss

        if mode == 'test':
            similarities = self.compute_similarities(
                h, b_encoded, res_encoded, p_num, raw_timestamps,
                alpha_plus, alpha_minus, sim_plus, sim_minus,
                p_base, p_plus, p_minus, p_both, te_p, te_r
            )
            return total_loss, b_loss, res_loss, logits_both, logits_res, time_interval_for_attention, att_score, b_y, logits_b, res_y, te_loss, s_loss_raw, similarities
        else:
            return total_loss, b_loss, res_loss, logits_both, logits_res, time_interval_for_attention, att_score, b_y, logits_b, res_y, te_loss, s_loss_raw

    def compute_similarities(
        self, user_repr, b_factors, res_factors, p_num, raw_timestamps,
        alpha_plus, alpha_minus, sim_plus, sim_minus,
        p_base, p_plus, p_minus, p_both, te_p, te_r
    ):
        """Compute per-post factor similarities, weights, and TE values for test-time analysis."""
        batch_size = user_repr.shape[0]

        similarities = {
            'factor_similarities': [],
            'factor_weights': [],
            'factor_types': [],
            'p_base_values': [],
            'p_plus_values': [],
            'p_minus_values': [],
            'p_both_values': [],
            'te_p_values': [],
            'te_r_values': [],
            'te_p_sigmoid': [],
            'te_r_sigmoid': [],
            'timestamps': [],
            'post_indices': []
        }

        for i in range(batch_size):
            actual_posts = min(p_num[i].item(), b_factors.shape[1])
            if actual_posts == 0:
                for key in ['factor_similarities', 'factor_weights', 'factor_types', 'timestamps', 'post_indices']:
                    similarities[key].append([])
                for key in ['p_base_values', 'p_plus_values', 'p_minus_values', 'p_both_values', 'te_p_values', 'te_r_values']:
                    similarities[key].append(0.0)
                similarities['te_p_sigmoid'].append(0.5)
                similarities['te_r_sigmoid'].append(0.5)
                continue

            user_raw_timestamps = raw_timestamps[i][:actual_posts]
            readable_timestamps = [str(ts) for ts in user_raw_timestamps]

            user_alpha_plus = alpha_plus[i, :actual_posts]
            user_alpha_minus = alpha_minus[i, :actual_posts]
            user_sim_plus = sim_plus[i, :actual_posts]
            user_sim_minus = sim_minus[i, :actual_posts]

            all_factor_similarities = []
            all_factor_weights = []
            all_factor_types = []
            all_timestamps_expanded = []
            all_post_indices = []

            for post_idx in range(actual_posts):
                all_factor_similarities.append(user_sim_plus[post_idx].item())
                all_factor_weights.append(user_alpha_plus[post_idx].item())
                all_factor_types.append('protection')
                all_timestamps_expanded.append(readable_timestamps[post_idx])
                all_post_indices.append(post_idx)

                all_factor_similarities.append(user_sim_minus[post_idx].item())
                all_factor_weights.append(user_alpha_minus[post_idx].item())
                all_factor_types.append('risk')
                all_timestamps_expanded.append(readable_timestamps[post_idx])
                all_post_indices.append(post_idx)

            similarities['factor_similarities'].append(all_factor_similarities)
            similarities['factor_weights'].append(all_factor_weights)
            similarities['factor_types'].append(all_factor_types)
            similarities['p_base_values'].append(p_base[i].item())
            similarities['p_plus_values'].append(p_plus[i].item())
            similarities['p_minus_values'].append(p_minus[i].item())
            similarities['p_both_values'].append(p_both[i].item())
            similarities['te_p_values'].append(te_p[i].item())
            similarities['te_r_values'].append(te_r[i].item())
            similarities['te_p_sigmoid'].append(float(torch.sigmoid(self.gamma_te * te_p[i]).item()))
            similarities['te_r_sigmoid'].append(float(torch.sigmoid(self.gamma_te * te_r[i]).item()))
            similarities['timestamps'].append(all_timestamps_expanded)
            similarities['post_indices'].append(all_post_indices)

        return similarities

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.001)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def aggregate_factors(self, user_repr, factors, lengths, tau):
        """Softmax-weighted aggregation of factor representations by cosine similarity to user repr."""
        batch_size, max_len, hidden_dim = factors.shape
        device = factors.device
        lengths = lengths.to(device)
        eps = 1e-8

        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        u = user_repr.unsqueeze(1).expand(-1, max_len, -1)

        numerator = (u * factors).sum(dim=-1)
        u_norm = torch.norm(u, dim=-1)
        f_norm = torch.norm(factors, dim=-1)
        sim = numerator / (u_norm * f_norm + eps)
        sim = sim / tau
        sim = sim.masked_fill(~mask, -1e9)

        weights = F.softmax(sim, dim=1)
        if (lengths == 0).any():
            zero_mask = lengths == 0
            weights = weights.clone()
            weights[zero_mask] = 0.0

        agg = torch.bmm(weights.unsqueeze(1), factors).squeeze(1)
        return agg, weights, sim

    def preprocess_dataframe(self):
        data_path = self.config.get('dataset_path', './dataset/data_new_window_size_4.pkl')
        if str(data_path).lower().endswith((".parquet", ".pq")):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_pickle(data_path)

        self.s_y_col = "fu_" + str(self.config['af']) + "_su_y"
        if self.config['s_y_num'] == 3:
            df[self.s_y_col] = df[self.s_y_col].apply(lambda x: 2 if x in [2, 3] else x)
        elif self.config['s_y_num'] == 2:
            df[self.s_y_col] = df[self.s_y_col].apply(lambda x: 1 if x in [1, 2, 3] else x)

        split_seed = self.config.get('split_seed', self.config['random_seed'])
        val_ratio = self.config.get('val_ratio', 0.0)
        fold_idx = self.config.get('n_fold_index', self.config['n_fold'])
        self.df_train, self.df_val, self.df_test = split_folds(
            df=df,
            label_col=self.s_y_col,
            group_col='author',
            n_splits=5,
            fold_idx=fold_idx,
            seed=split_seed,
            val_ratio=val_ratio,
        )

        # Within-fold RF feature selection.
        # Fit exclusively on training data; apply the learned indices to val/test.
        # This prevents any information leakage from held-out splits.
        if self.config.get('rf_feature_selection', True):
            n_risk = self.config.get('b_y_num', 4)
            n_protective = self.config.get('res_y_num', 4)
            seed = self.config.get('random_seed', 42)
            risk_idx, protective_idx = select_factors_rf(
                self.df_train, self.s_y_col, n_risk, n_protective, seed
            )
            self.df_train = apply_factor_selection(self.df_train, risk_idx, protective_idx)
            if not self.df_val.empty:
                self.df_val = apply_factor_selection(self.df_val, risk_idx, protective_idx)
            self.df_test = apply_factor_selection(self.df_test, risk_idx, protective_idx)
            self.selected_risk_indices = risk_idx
            self.selected_protective_indices = protective_idx

    def train_dataloader(self):
        self.train_data = RedditDataset(
            self.df_train[self.s_y_col].values,
            self.df_train['cur_su_y'].tolist(),
            self.df_train['cur_bp_y'].values,
            self.df_train['cur_bp_res'].values,
            self.df_train[self.embed_type].values,
            self.df_train["created_utc"].values,
            self.df_train['user_id'].values
        )
        return DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            collate_fn=pad_collate_reddit,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )

    def val_dataloader(self):
        self.val_data = RedditDataset(
            self.df_val[self.s_y_col].values,
            self.df_val['cur_su_y'].tolist(),
            self.df_val['cur_bp_y'].values,
            self.df_val['cur_bp_res'].values,
            self.df_val[self.embed_type].values,
            self.df_val["created_utc"].values,
            self.df_val['user_id'].values
        )
        return DataLoader(
            self.val_data,
            batch_size=self.args.batch_size,
            collate_fn=pad_collate_reddit,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )

    def test_dataloader(self):
        self.test_data = RedditDataset(
            self.df_test[self.s_y_col].values,
            self.df_test['cur_su_y'].tolist(),
            self.df_test['cur_bp_y'].values,
            self.df_test['cur_bp_res'].values,
            self.df_test[self.embed_type].values,
            self.df_test["created_utc"].values,
            self.df_test['user_id'].values
        )
        return DataLoader(
            self.test_data,
            batch_size=self.args.batch_size,
            collate_fn=pad_collate_reddit,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )

    def training_step(self, batch, batch_idx):
        s_y, cur_su_y, b_y, res_y, p_num, tweets, time_interval, raw_timestamps, user_id = batch
        loss, b_loss, res_loss, logit_s, logit_res, time_interval, att_score, b_true, b_pred, res_y, te_loss, s_loss_raw = self(
            cur_su_y, s_y, b_y, res_y, p_num, tweets, time_interval, raw_timestamps=None, mode='train'
        )
        self.log("train_loss", loss)
        self.log("train_s_loss_raw", s_loss_raw)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        s_y, cur_su_y, b_y, res_y, p_num, tweets, time_interval, raw_timestamps, user_id = batch
        loss, b_loss, res_loss, logit_s, logit_res, time_interval, att_score, b_true, b_pred, res_y, te_loss, s_loss_raw = self(
            cur_su_y, s_y, b_y, res_y, p_num, tweets, time_interval, raw_timestamps=None, mode='train'
        )
        self.log("val_loss", s_loss_raw, prog_bar=True)
        return {'val_loss': s_loss_raw}

    def test_step(self, batch, batch_idx):
        s_y, cur_su_y, b_y, res_y, p_num, tweets, time_interval, raw_timestamps, user_id = batch
        loss, b_loss, res_loss, logit_s, logit_res, time_interval, att_score, b_true, b_pred, res_y, te_loss, s_loss_raw, similarities = self(
            cur_su_y, s_y, b_y, res_y, p_num, tweets, time_interval, raw_timestamps, mode='test'
        )

        temporal_att_weights = att_score.detach().cpu().numpy()
        processed_temporal_att = []
        for i in range(len(p_num)):
            actual_posts = min(p_num[i].item(), temporal_att_weights.shape[1])
            user_att_weights = temporal_att_weights[i, :actual_posts].tolist()
            processed_temporal_att.append(user_att_weights)
        similarities['temporal_att_weights'] = processed_temporal_att

        s_true = list(s_y.cpu().numpy())
        s_preds = list(logit_s.argmax(dim=-1).cpu().numpy())

        b_true = list(b_true.cpu().numpy())
        b_pred = F.softmax(b_pred, dim=1)
        b_preds = np.array(b_pred.cpu() > 0.14).astype(int)
        b_preds = list(b_preds)

        res_true = list(res_y.cpu().numpy())
        res_pred = F.softmax(logit_res, dim=1)
        res_preds = np.array(res_pred.cpu() > 0.14).astype(int)
        res_preds = list(res_preds)

        user_id = list(user_id.cpu().numpy())

        return {
            'loss': loss,
            's_true': s_true,
            's_preds': s_preds,
            'b_true': b_true,
            'b_preds': b_preds,
            'res_true': res_true,
            'res_preds': res_preds,
            'user_id': user_id,
            'similarities': similarities
        }

    def test_epoch_end(self, outputs):
        if outputs:
            avg_loss = torch.stack([o['loss'] for o in outputs]).mean()
            self.log("test_loss", avg_loss, prog_bar=True)
        evaluation(self.config, outputs, 'fs', 's_true', 's_preds', 'user_id')
        evaluation(self.config, outputs, 'bd', 'b_true', 'b_preds', 'user_id')
        evaluation(self.config, outputs, 'res', 'res_true', 'res_preds', 'user_id')
        if outputs:
            s_true = np.asanyarray([v for o in outputs for v in o['s_true']])
            s_preds = np.asanyarray([v for o in outputs for v in o['s_preds']])
            gp, gr, fs, _ = gr_metrics(s_preds, s_true)
            report = classification_report(
                s_true,
                s_preds,
                zero_division=1,
                target_names=['su_indicator', 'su_ideation', 'su_behavior', 'su_attempt'],
                output_dict=True,
            )
            weighted = report.get('weighted avg', {})
            self.log("weighted_precision", float(weighted.get('precision', 0.0)))
            self.log("weighted_recall", float(weighted.get('recall', 0.0)))
            self.log("weighted_f1", float(weighted.get('f1-score', 0.0)))
            self.log("GP", float(gp))
            self.log("GR", float(gr))
            self.log("FS", float(fs))
