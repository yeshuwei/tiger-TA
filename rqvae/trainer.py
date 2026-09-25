import logging
from collections import defaultdict

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir, set_color, get_local_time, delete_file
from trans_loss import ContrastiveLoss, BalanceLoss
import os
import heapq


class Trainer(object):

    def __init__(self, args, model, data_num):
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

    def _build_optimizer(self):
        params = self.model.parameters()
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

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_trans_loss = 0.0
        total_balance_loss = 0.0

        is_track_b = (self.lambda_trans > 0 or self.lambda_balance > 0)

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
                out, rq_loss, vq_indices, z = self.model(data, return_z=True)
            else:
                # Track A: 3-value return for backward compatibility
                out, rq_loss, vq_indices = self.model(data, return_z=False)

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

            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            if is_track_b:
                total_trans_loss += loss_trans.item() if torch.is_tensor(loss_trans) else 0.0
                total_balance_loss += loss_balance.item() if torch.is_tensor(loss_balance) else 0.0

        # Log extra losses for Track B
        if is_track_b:
            if total_trans_loss > 0 or total_balance_loss > 0:
                n = len(iter_data)
                self.logger.info(
                    f"    L_recon={total_recon_loss/n:.4f}, "
                    f"L_trans={total_trans_loss/n:.4f}, "
                    f"L_balance={total_balance_loss/n:.4f}"
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
