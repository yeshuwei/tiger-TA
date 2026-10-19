import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir, set_color, get_local_time, delete_file
from trans_loss import ContrastiveLoss, BalanceLoss
import os
import heapq


class Trainer(object):

    def __init__(self, args, model, data_num, csg=None):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"

        # ---- E50 b2: L_align ----------------------------------------------
        # 必须在 _build_optimizer() **之前**建好，否则 W 进不了优化器。
        # align_weight == 0 时 self.align_proj is None，下面每条路径（optimizer
        # 参数、前向返回、clip_grad、ckpt）都与改 b2 之前的 trainer 逐位一致
        # —— 这是 doc/exp/ab_decision_and_design.md §4.1 尾注的回归门禁。
        self.align_weight = getattr(args, 'align_weight', 0.0)
        self.align_tau = getattr(args, 'align_tau', 0.1)
        self.align_proj = None
        self.align_table = None
        self.align_num_dropped = 0
        if self.align_weight > 0:
            align_path = getattr(args, 'align_data_path', None)
            if not align_path or not os.path.exists(align_path):
                raise FileNotFoundError(
                    f"align_weight={self.align_weight} 但 align_data_path 不可用: {align_path}"
                )
            align_df = pd.read_parquet(align_path)
            align_np = np.stack(align_df['embedding'].values).astype(np.float32)
            # 常驻 CPU，按 batch 行号 gather 后再搬上 device（见 _get_align_loss）
            self.align_table = torch.from_numpy(align_np)
            # ⚠️ 投影头是**维度所迫**，不是设计选择：x_q 只有 e_dim(=32) 维，
            #    而目标 h 是 128 维，不加投影 ⟨x_q, h⟩ 无量纲定义。
            self.align_proj = nn.Linear(model.e_dim, align_np.shape[1], bias=False).to(self.device)
            self.logger.info(
                f"[L_align] weight={self.align_weight} tau={self.align_tau} "
                f"target={align_path} dim={align_np.shape[1]} "
                f"proj={model.e_dim}->{align_np.shape[1]}"
            )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        # ---- Track B: L_trans + L_balance ----
        self.lambda_trans = getattr(args, 'lambda_trans', 0.0)
        self.lambda_balance = getattr(args, 'lambda_balance', 0.0)
        self.tau = getattr(args, 'tau', 0.1)

        # Contrastive loss module
        self.contrastive_loss = ContrastiveLoss(tau=self.tau).to(self.device)

        # Balance loss module
        self.balance_loss = BalanceLoss()
        self.num_emb_list = args.num_emb_list

        # Load transition pairs if lambda_trans > 0
        self.trans_dict = None  # item_idx → list of positive item indices
        if self.lambda_trans > 0:
            trans_path = getattr(args, 'trans_graph_path', None)
            if trans_path and os.path.exists(trans_path):
                self.logger.info(f"Loading transition pairs from {trans_path}")
                trans_pairs = np.load(trans_path)  # (N, 2)
                self.trans_dict = defaultdict(list)
                for a, b in trans_pairs:
                    self.trans_dict[a].append(b)
                num_pairs = len(trans_pairs)
                unique_items = len(self.trans_dict)
                self.logger.info(f"  Loaded {num_pairs} pairs, {unique_items} unique items")
                # Free memory
                del trans_pairs
            else:
                self.logger.warning(
                    f"lambda_trans={self.lambda_trans} but trans_graph_path not found: {trans_path}"
                )
                self.logger.warning("Disabling L_trans")
                self.lambda_trans = 0.0

        # ---- C5: the Q-space constraint (lambda_csg, default 0) -------------
        # A parameter-free hook built by ``tiger_ext/qspace_train.py`` that holds
        # C3's constraint and C4's objective and returns ``lambda_csg * L_csg``.
        # ``csg is None`` -- the default, and what every existing caller passes --
        # means every expression added for it below evaluates exactly as before
        # and no extra tensor is created or read, so the historical behaviour is
        # bitwise unchanged.
        self.csg = csg

    def _build_optimizer(self):
        params = self.model.parameters()
        if getattr(self, 'align_proj', None) is not None:
            # 只有 b2 才把投影头加进来；align_weight=0 时走原样路径
            params = list(params) + list(self.align_proj.parameters())
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
            )
        return lr_scheduler

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _get_transition_loss(self, vq_indices, batch_indices):
        """Compute L_trans for in-batch transition pairs, applied on quantized vectors (x_q).

        ╔══════════════════════════════════════════════════════════════╗
        ║  v2: L_trans on x_q (quantized) instead of z (encoder).    ║
        ║  Why: z-based L_trans fights the quantizer (pulls encoder   ║
        ║  output together → quantizer spreads apart → code quality   ║
        ║  suffers). x_q-based L_trans aligns with the discrete code  ║
        ║  representation that T5 actually consumes.                  ║
        ╚══════════════════════════════════════════════════════════════╝

        Args:
            vq_indices: (B, num_layers) quantized code indices (each 0..255)
            batch_indices: (B,) item indices in this batch

        Returns:
            scalar loss (0 if no transition pairs found in batch)
        """
        B = vq_indices.shape[0]
        if self.trans_dict is None or B < 2:
            return torch.tensor(0.0, device=vq_indices.device, dtype=torch.float32)

        # Reconstruct quantized vectors from codebook indices
        # x_q = Σ_l codebook_l[vq_indices[:, l]] — same as rq.forward() builds
        num_layers = vq_indices.shape[1]
        x_q = torch.zeros(B, self.model.e_dim, device=vq_indices.device)
        for layer in range(num_layers):
            codebook_w = self.model.rq.vq_layers[layer].embedding.weight  # (256, e_dim)
            x_q = x_q + codebook_w[vq_indices[:, layer]]                  # (B, e_dim)

        # Build batch position → item idx mapping
        idx_to_pos = {}
        for pos, item_idx in enumerate(batch_indices):
            item_idx_int = item_idx.item() if torch.is_tensor(item_idx) else item_idx
            idx_to_pos[item_idx_int] = pos

        # Find transition pairs within this batch
        anchors = []
        positives = []

        for pos, item_idx in enumerate(batch_indices):
            item_idx_int = item_idx.item() if torch.is_tensor(item_idx) else item_idx
            if item_idx_int in self.trans_dict:
                for target_idx in self.trans_dict[item_idx_int]:
                    if target_idx in idx_to_pos:
                        anchors.append(pos)
                        positives.append(idx_to_pos[target_idx])

        if len(anchors) == 0:
            return torch.tensor(0.0, device=vq_indices.device, dtype=torch.float32)

        anchor_tensor = torch.tensor(anchors, dtype=torch.long, device=vq_indices.device)
        pos_tensor = torch.tensor(positives, dtype=torch.long, device=vq_indices.device)

        return self.contrastive_loss(x_q, anchor_tensor, pos_tensor)

    def _get_align_loss(self, x_q, batch_indices):
        """E50 b2: InfoNCE 对齐 —— 把量化表示拉向目标嵌入（LETTER eq.3 形式）。

            L_align = -1/B Σ_i log[ exp(⟨w_i, h_i⟩/τ) / Σ_j exp(⟨w_i, h_j⟩/τ) ]

        其中（同 batch 内互为负样本）：
            w_i = normalize(W · x_q,i)     W: e_dim(32) → D(128)，可学习
            h_i = normalize(h_i)           h = 目标嵌入的第 i 行（不传梯度）

        ⚠️ 两个必须注意的点（见 doc/exp/ab_decision_and_design.md §4.2）：
          A) x_q 只有 e_dim=32 维，h 是 128 维 ⇒ 投影 W 不可省，否则点积无定义。
          C) 有一批物品的行为向量恒为零（无转移记录的冷启动物品，实测
             Sport/Toys/Beauty = 28/24/33 行）。normalize 零向量会放大成纯噪声，
             故按 ‖h‖ < 1e-6 直接 mask 掉，不计入损失。

        Args:
            x_q: (B, e_dim) 模型内部的量化表示（**必须**取自 forward(return_xq=True)，
                 它带有经 straight-through 回流到 encoder 的梯度）
            batch_indices: (B,) 该 batch 在 parquet 中的行号

        Returns:
            scalar loss（无有效行时返回 0）
        """
        zero = torch.tensor(0.0, device=x_q.device, dtype=torch.float32)
        if self.align_table is None or self.align_proj is None or x_q is None:
            return zero

        B = x_q.shape[0]
        if B < 2:
            return zero

        # 按行号 gather 目标嵌入（EmbDataset 的 index 就是 parquet 行号）
        idx = batch_indices
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(list(idx), dtype=torch.long)
        idx = idx.detach().to(device=self.align_table.device, dtype=torch.long)
        h = self.align_table[idx].to(x_q.device)                          # (B, D)

        # 丢掉退化行（h == 0 的冷启动物品）
        keep = h.norm(dim=1) > 1e-6
        n_keep = int(keep.sum().item())
        self.align_num_dropped += B - n_keep
        if n_keep < 2:
            return zero
        x_q = x_q[keep]
        h = h[keep]

        w = F.normalize(self.align_proj(x_q), dim=1)                      # (B', D)
        h = F.normalize(h.detach(), dim=1)                                # (B', D)，目标不传梯度

        logits = w @ h.t() / self.align_tau
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_trans_loss = 0.0
        total_balance_loss = 0.0
        total_align_loss = 0.0
        total_csg = 0.0

        is_track_b = (self.lambda_trans > 0 or self.lambda_balance > 0
                      or self.align_weight > 0
                      or (self.csg is not None and self.csg.enabled))

        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
        )

        for batch_idx, batch in enumerate(iter_data):
            if is_track_b:
                # Dataset returns (idx, embedding) tuples
                batch_indices, data = batch
                data = data.to(self.device)
            else:
                data = batch.to(self.device)
                batch_indices = None

            self.optimizer.zero_grad()

            # Forward: get reconstruction + latent z (if Track B)
            if is_track_b:
                if self.align_weight > 0 or (self.csg is not None and self.csg.enabled):
                    # b2 / C5 额外取 x_q —— 必须是模型内部那条 straight-through 的
                    # 张量，而非外部重新 gather（否则梯度到不了 encoder）；
                    # CSG 还要 z，用于 Q-space provenance 校验（x_q != z）
                    out, rq_loss, vq_indices, z, x_q = self.model(
                        data, return_z=True, return_xq=True)
                else:
                    out, rq_loss, vq_indices, z = self.model(data, return_z=True)
                    x_q = None
            else:
                # Track A: 3-value return for backward compatibility
                out, rq_loss, vq_indices = self.model(data, return_z=False)
                x_q = None

            # Base loss (reconstruction + VQ commitment)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)

            # L_trans (v2: applied on quantized vectors x_q, not encoder output z)
            loss_trans = torch.tensor(0.0, device=self.device)
            if self.lambda_trans > 0 and self.trans_dict is not None and batch_indices is not None:
                loss_trans = self._get_transition_loss(vq_indices, batch_indices)
                if loss_trans.requires_grad and torch.isfinite(loss_trans):
                    loss = loss + self.lambda_trans * loss_trans

            # L_balance
            loss_balance = torch.tensor(0.0, device=self.device)
            if self.lambda_balance > 0:
                # vq_indices: (B, num_layers) stacked tensor
                loss_balance = self.balance_loss(vq_indices, self.num_emb_list)
                if loss_balance.requires_grad and torch.isfinite(loss_balance):
                    loss = loss + self.lambda_balance * loss_balance

            # L_align (E50 b2)
            loss_align = torch.tensor(0.0, device=self.device)
            if self.align_weight > 0 and batch_indices is not None:
                loss_align = self._get_align_loss(x_q, batch_indices)
                if loss_align.requires_grad and torch.isfinite(loss_align):
                    loss = loss + self.align_weight * loss_align

            # L_csg (C5) -- the hook already returns lambda_csg * L_csg, exactly
            # like SIAMount, so the weight is applied once, inside the hook.
            loss_csg = torch.tensor(0.0, device=self.device)
            if self.csg is not None and self.csg.enabled and batch_indices is not None:
                loss_csg = self.csg(x_q=x_q, indices=vq_indices, latent=z,
                                    rows=batch_indices, rqvae=self.model)
                if loss_csg.requires_grad and torch.isfinite(loss_csg):
                    loss = loss + loss_csg

            self._check_nan(loss)
            loss.backward()
            clip_params = self.model.parameters()
            if self.align_proj is not None:
                clip_params = list(clip_params) + list(self.align_proj.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            if is_track_b:
                total_trans_loss += loss_trans.item() if torch.is_tensor(loss_trans) else 0.0
                total_balance_loss += loss_balance.item() if torch.is_tensor(loss_balance) else 0.0
                total_align_loss += loss_align.item() if torch.is_tensor(loss_align) else 0.0
                total_csg += loss_csg.item() if torch.is_tensor(loss_csg) else 0.0

        # Log extra losses
        if is_track_b:
            if total_trans_loss > 0 or total_balance_loss > 0 or total_align_loss > 0:
                n = len(iter_data)
                self.logger.info(
                    f"    L_recon={total_recon_loss/n:.4f}, "
                    f"L_trans={total_trans_loss/n:.4f}, "
                    f"L_balance={total_balance_loss/n:.4f}, "
                    f"L_align={total_align_loss/n:.4f}"
                )

        if self.csg is not None and self.csg.enabled:
            n = len(iter_data)
            self.logger.info(
                f"    L_recon={total_recon_loss/n:.4f}, "
                f"L_csg={total_csg/n:.4f}"
            )

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
        )

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            if isinstance(data, (list, tuple)):
                # Track B: (indices, embeddings)
                data = data[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set))) / num_sample
        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        # E50 b2: 投影头只存为**顶层键**，不进 state_dict —— 既有 ckpt 读取方
        # （model.load_state_dict(ck["state_dict"])）完全不受影响。
        if self.align_proj is not None:
            state["align_proj"] = self.align_proj.state_dict()
            state["align_meta"] = {
                "align_weight": self.align_weight,
                "align_tau": self.align_tau,
                "align_data_path": getattr(self.args, 'align_data_path', None),
            }
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")
        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output += ", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"

    def fit(self, data):
        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(
                        epoch_idx, collision_rate=collision_rate,
                        ckpt_file=self.best_collision_ckpt,
                    )
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)
                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])
                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        return self.best_loss, self.best_collision_rate
