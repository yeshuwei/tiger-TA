#!/usr/bin/env python3
"""
run_checkpoint_trajectory.py — Checkpoint Trajectory 实验的训练驱动

对应实验方案：`doc/exp/checkpoint_trajectory/experiment_plan.md`

目的：在**同一次 training run** 内、于**固定训练轮次**（默认 50/500/1000/2000）保存
checkpoint，供 `tools/checkpoint_trajectory_probe.py` 比较不同 epoch 的
behavior structure preservation。**禁止**用 best_loss / best_collision 作为取点依据。

设计要点（为什么不新写 trainer）：
  - `from main import parse_args` —— 逐字复用 `rqvae/main.py` 的超参默认值，
    保证与既有 9 个 TA-SID run 完全同构（epochs3000 / batch1024 / lr1e-3 / 256³ /
    sk_epsilons[0,0,0.003] / warmup50 / AdamW / linear）。
  - 播种块与 `main.py` 的 `__main__` 逐字一致（含 cudnn.deterministic）。
  - `TrajectoryTrainer` **只重写两个方法**，`trainer.py` 一字不改：
        `_save_checkpoint`  → 在周期性保存点（`ckpt_file is None`）额外落一份 traj ckpt
        `_generate_train_loss_output` → 顺手记下该 epoch 的 train/recon loss 供入档
    这两处都不消耗 RNG、不改梯度、不改任何保存/比较逻辑 ⇒ 训练轨迹与 `main.py` 逐位相同。
  - epoch 语义：trainer 只在 `(epoch_idx + 1) % eval_step == 0` 时评估/保存，
    故「完成 N 轮训练」对应 `epoch_idx = N - 1`（N=50 → 49）。`epoch_label = epoch_idx + 1`。
    周期性 `_save_checkpoint(epoch_idx, collision_rate=...)` 发生在 eval 之后、
    下一轮训练之前 ⇒ 快照正是「完成 epoch_idx+1 轮训练后」的状态。

用法（仓库根，与 main.py 同参，另加 --traj_epochs）：
    python rqvae/run_checkpoint_trajectory.py \
        --data_path data/Toys/fused_item_emb_768d.parquet \
        --ckpt_dir rqvae/checkpoint_trajectory/ckpt/Toys_TA_SID_seed2026 \
        --seed 2026 \
        --traj_epochs 50 500 1000 2000

输出：`{ckpt_dir}/{timestamp}/traj_epoch{N:05d}.pth`（N=50/500/1000/2000），
      以及原有的 best_loss / best_collision / epoch_* 文件（行为不变）。
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import EmbDataset                      # noqa: E402
from models.rqvae import RQVAE                        # noqa: E402
from trainer import Trainer                           # noqa: E402
from utils import set_color                           # noqa: E402
from main import parse_args                           # noqa: E402

DEFAULT_TRAJ_EPOCHS = [50, 500, 1000, 2000]


class TrajectoryTrainer(Trainer):
    """Trainer 的只增不改子类：在固定 epoch 的评估点额外留一份 trajectory checkpoint。"""

    def __init__(self, args, model, data_num, traj_epochs):
        super().__init__(args, model, data_num)
        self.traj_epochs = set(int(e) for e in traj_epochs)
        self._last_train_loss = float("nan")
        self._last_train_recon_loss = float("nan")
        self.traj_saved = []

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        # 只做记录，不改返回值/行为
        self._last_train_loss = float(loss)
        self._last_train_recon_loss = float(recon_loss)
        return super()._generate_train_loss_output(epoch_idx, s_time, e_time, loss, recon_loss)

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):
        path = super()._save_checkpoint(epoch, collision_rate, ckpt_file)
        # 只在**周期性**保存点（ckpt_file is None）额外落 traj ckpt；
        # best_loss / best_collision 那两次调用不触发（保持原逻辑与文件集合不变）。
        if ckpt_file is None and (epoch + 1) in self.traj_epochs:
            traj_path = os.path.join(self.ckpt_dir, "traj_epoch%05d.pth" % (epoch + 1))
            state = {
                "args": self.args,
                "epoch": epoch,                 # epoch_idx（0-based）
                "epoch_label": epoch + 1,       # 完成训练的轮数 = 50/500/1000/2000
                "best_loss": self.best_loss,
                "best_collision_rate": self.best_collision_rate,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                # 该 epoch 的训练日志值（trainer 口径），供审计时与 log 逐行对照
                "train_loss": self._last_train_loss,
                "train_recon_loss": self._last_train_recon_loss,
                "valid_collision_rate": float(collision_rate),
                "trajectory": True,
            }
            torch.save(state, traj_path, pickle_protocol=4)
            self.traj_saved.append(traj_path)
            self.logger.info(set_color("Saving trajectory", "blue") + f": {traj_path}")
        return path


def parse_extra(argv):
    """先摘出本实验专属参数，其余原样交给 main.parse_args()。"""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--traj_epochs", type=int, nargs="+", default=DEFAULT_TRAJ_EPOCHS)
    known, rest = pre.parse_known_args(argv)
    return known, rest


if __name__ == "__main__":
    extra, rest = parse_extra(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    args = parse_args()

    """fix the random seed —— 与 rqvae/main.py 逐字一致"""
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=================================================")
    print(args)
    print(f"traj_epochs = {sorted(set(extra.traj_epochs))}")
    print("=================================================")

    import logging
    logging.basicConfig(level=logging.DEBUG)   # 与 main.py 逐字一致

    """build dataset —— 与 main.py 逐字一致"""
    is_track_b = args.lambda_trans > 0 or args.lambda_balance > 0
    data = EmbDataset(args.data_path, return_index=is_track_b)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
    print(model)
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = TrajectoryTrainer(args, model, len(data_loader), extra.traj_epochs)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)
    print(f"Trajectory checkpoints ({len(trainer.traj_saved)}):")
    for p in trainer.traj_saved:
        print("   ", p)
