import os
import argparse
import numpy as np
import random
import warnings

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')

from src.TempATT import TempATT


def th_seed_everything(seed: int = 2023):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Arg:
    epochs: int = 200
    report_cycle: int = 30
    cpu_workers: int = os.cpu_count()
    test_mode: bool = False
    optimizer: str = 'AdamW'
    lr_scheduler: str = 'exp'
    fp16: bool = False
    batch_size: int = 64
    max_post_num = 30
    task_num: int = 0
    weight_decay: float = 0.01


def main(args, config):
    if "n_fold_index" not in config:
        config["n_fold_index"] = max(config.get("n_fold", 1) - 1, 0)
    split_seed = config.get('split_seed', config['random_seed'])
    random_seed = config.get('random_seed', split_seed)
    seed_everything(random_seed)
    th_seed_everything(random_seed)
    config['res_y_num'] = 4

    model = TempATT(args, config)
    model.preprocess_dataframe()

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.get('early_stop_patience', 10),
        verbose=True,
        mode='min'
    )
    save_root = config.get('save', 'checkpoints')
    fold_suffix = f"fold{config.get('n_fold', 0)}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_root, fold_suffix),
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=True,
        gpus=1,
        precision=16 if args.fp16 else 32
    )
    trainer.fit(model)
    test_results = trainer.test(model, dataloaders=model.test_dataloader())
    return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index")
    parser.add_argument("--random_seed", type=int, default=2022)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--bf", type=int, default=6)
    parser.add_argument("--af", type=int, default=30)
    parser.add_argument("--embed_type", type=str, default="sb")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--loss", type=str, default="oe")
    parser.add_argument("--save", type=str, default="test")
    parser.add_argument("--s_y_num", type=int, default=4)
    parser.add_argument("--b_y_num", type=int, default=4)
    parser.add_argument("--res_y_num", type=int, default=4, help="number of resilience factor classes")
    parser.add_argument("--n_fold", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.125)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--run_all_folds", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for AdamW")
    parser.add_argument("--lr_scheduler", type=str, default="exp", choices=["exp", "cosine"], help="learning rate scheduler")
    parser.add_argument("--tau", type=float, default=0.8, help="temperature for factor aggregation")
    parser.add_argument("--lambda_te", type=float, default=0.2, help="weight for TE loss")
    parser.add_argument("--gamma_te", type=float, default=1.0, help="gamma for TE sigmoid scaling")
    parser.add_argument("--lambda_sr", type=float, default=1.0, help="weight for suicide risk loss")
    parser.add_argument("--lambda_pf", type=float, default=0.4, help="weight for protective factor loss")
    parser.add_argument("--lambda_rf", type=float, default=0.1, help="weight for risk factor loss")
    parser.add_argument("--input", type=str, default=None, help="path to input dataset pkl")
    parser.add_argument(
        "--rf_feature_selection", action="store_true", default=True,
        help="perform RF feature selection strictly within each training fold"
    )
    parser.add_argument(
        "--no_rf_feature_selection", dest="rf_feature_selection", action="store_false",
        help="disable within-fold RF feature selection"
    )

    config = parser.parse_args()
    args = Arg()
    args.weight_decay = config.weight_decay
    args.lr_scheduler = config.lr_scheduler

    config = config.__dict__
    if config.get("input"):
        config["dataset_path"] = config["input"]

    if config.get('run_all_folds'):
        all_results = []
        weighted_precision_list = []
        weighted_recall_list = []
        weighted_f1_list = []
        gp_list = []
        gr_list = []
        fs_list = []
        for fold_num in range(1, 6):
            config['n_fold'] = fold_num
            config['n_fold_index'] = fold_num - 1
            print(f"Running fold {fold_num}/5")
            fold_results = main(args, config)
            all_results.append(fold_results)
            if isinstance(fold_results, list) and fold_results:
                metrics = fold_results[0]
                if 'weighted_precision' in metrics:
                    weighted_precision_list.append(metrics['weighted_precision'])
                if 'weighted_recall' in metrics:
                    weighted_recall_list.append(metrics['weighted_recall'])
                if 'weighted_f1' in metrics:
                    weighted_f1_list.append(metrics['weighted_f1'])
                if 'GP' in metrics:
                    gp_list.append(metrics['GP'])
                if 'GR' in metrics:
                    gr_list.append(metrics['GR'])
                if 'FS' in metrics:
                    fs_list.append(metrics['FS'])

        test_losses = []
        for fold_results in all_results:
            if isinstance(fold_results, list) and fold_results:
                test_loss = fold_results[0].get('test_loss')
                if test_loss is not None:
                    test_losses.append(test_loss)

        if test_losses:
            mean_loss = float(np.mean(test_losses))
            std_loss = float(np.std(test_losses, ddof=1)) if len(test_losses) > 1 else 0.0
            print(f"5-fold test_loss: {mean_loss:.6f} ± {std_loss:.6f}")
        else:
            print("No test_loss collected from folds.")

        if weighted_precision_list:
            mean_wp = float(np.mean(weighted_precision_list))
            std_wp = float(np.std(weighted_precision_list, ddof=1)) if len(weighted_precision_list) > 1 else 0.0
            mean_wr = float(np.mean(weighted_recall_list))
            std_wr = float(np.std(weighted_recall_list, ddof=1)) if len(weighted_recall_list) > 1 else 0.0
            mean_wf = float(np.mean(weighted_f1_list))
            std_wf = float(np.std(weighted_f1_list, ddof=1)) if len(weighted_f1_list) > 1 else 0.0
            mean_gp = float(np.mean(gp_list))
            std_gp = float(np.std(gp_list, ddof=1)) if len(gp_list) > 1 else 0.0
            mean_gr = float(np.mean(gr_list))
            std_gr = float(np.std(gr_list, ddof=1)) if len(gr_list) > 1 else 0.0
            mean_fs = float(np.mean(fs_list))
            std_fs = float(np.std(fs_list, ddof=1)) if len(fs_list) > 1 else 0.0
            print(
                "5-fold metrics: "
                f"weighted_precision {mean_wp:.6f} ± {std_wp:.6f}, "
                f"weighted_recall {mean_wr:.6f} ± {std_wr:.6f}, "
                f"weighted_f1 {mean_wf:.6f} ± {std_wf:.6f}, "
                f"GP {mean_gp:.6f} ± {std_gp:.6f}, "
                f"GR {mean_gr:.6f} ± {std_gr:.6f}, "
                f"FS {mean_fs:.6f} ± {std_fs:.6f}"
            )
    else:
        config["n_fold_index"] = max(config.get("n_fold", 1) - 1, 0)
        print(f"Running fold {config.get('n_fold', 1)}/5")
        main(args, config)
