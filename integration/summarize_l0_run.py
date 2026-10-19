#!/usr/bin/env python3
"""Summarize one EXP-L0 arm's ``run.json`` without touching the original.

Reads ``<arm>/run/run.json`` (produced by ``tiger_ext.fixed_train``), verifies
its structure, and writes a compact ``run_summary.json`` next to it (or at
``--out``).  Every value is copied verbatim from the source; fields that do
not exist in the source are reported as ``null`` and listed in
``missing_fields`` -- nothing is inferred or filled in.

Usage (from the repository root):

    python -I integration/summarize_l0_run.py \
        --run-json integration/artifacts/exp_l0/runs/<run-id>/<arm>/run/run.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

MISSING = object()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(obj, dotted: str, default=None):
    """Read ``a.b[0].c``; return ``default`` when any step is absent."""
    cur = obj
    for part in dotted.split("."):
        if "[" in part and part.endswith("]"):
            name, idx = part[:-1].split("[")
            if not isinstance(cur, dict) or name not in cur:
                return default
            cur = cur[name]
            try:
                cur = cur[int(idx)]
            except (IndexError, KeyError, TypeError):
                return default
        else:
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
    return cur


class Builder:
    """Accumulates the summary, the field-path map and the missing list."""

    def __init__(self, doc):
        self.doc = doc
        self.paths = {}
        self.missing = []

    def value(self, path, *, required=True, note=None):
        value = get(self.doc, path, MISSING)
        if value is MISSING:
            if required:
                self.missing.append({"field": path, "reason": "absent in run.json",
                                     **({"note": note} if note else {})})
            return None
        self.paths[path] = path
        return value

    def put(self, target: dict, key: str, path: str, *, note=None):
        value = self.value(path, note=note)
        target[key] = value
        return value

    def missing_field(self, name, *, note):
        self.missing.append({"field": name, "reason": "absent in run.json",
                             "note": note})
        return None


def build_summary(doc: dict, source_path: str) -> dict:
    if "config" not in doc or "result" not in doc:
        raise SystemExit(f"unexpected run.json shape: top-level keys = {list(doc)}")

    cfg = doc["config"]
    res = doc["result"]
    b = Builder(doc)

    best_epoch = get(res, "best_epoch", MISSING)
    epochs = get(res, "epochs", [])
    best_record = None
    if best_epoch is not MISSING and isinstance(epochs, list):
        best_record = next((e for e in epochs if e.get("epoch") == best_epoch), None)

    # ---- 1. experiment configuration -------------------------------------
    experiment = {}
    for key, path in (
        ("arm", "config.arm"),
        ("dataset", "config.dataset"),
        ("seed", "config.seed"),
        ("lr", "config.lr"),
        ("batch_size", "config.batch_size"),
        ("num_epochs_requested", "config.num_epochs"),
        ("early_stop", "config.early_stop"),
        ("lambda_sia", "config.lambda_sia"),
        ("lambda_psa", "config.lambda_psa"),
    ):
        b.put(experiment, key, path)
    experiment["lambda_csg"] = b.missing_field(
        "config.lambda_csg",
        note="not recorded in fixed_train run.json (CSG weight belongs to the "
             "tokenizer run; see <arm>/ckpt/*/ args or <arm>/refresh_manifest.json)")

    # ---- 2. data and model provenance ------------------------------------
    data_model = {}
    for key, path in (
        ("n_items", "config.n_items"),
        ("embedding_path", "config.embedding_path"),
        ("embedding_sha256", "config.embedding_sha256"),
        ("embedding_unchanged", "result.embedding_unchanged"),
        ("embedding_sha256_after", "result.embedding_sha256_after"),
        ("sid_path", "config.sid_path"),
        ("sid_sha256", "config.sid_sha256"),
        ("sid_manifest", "config.sid_manifest"),
        ("sid_provenance_mode", "config.sid_provenance_mode"),
        ("sid_provenance_verified", "config.sid_provenance.verified"),
        ("sid_version", "config.sid_provenance.version"),
        ("sid_unchanged", "result.sid_unchanged"),
        ("sid_sha256_after", "result.sid_sha256_after"),
        ("tokenizer_ckpt", "config.tokenizer_ckpt"),
        ("tokenizer_ckpt_sha256", "config.tokenizer_ckpt_sha256"),
        ("tokenizer_input_dim", "config.tokenizer_input_dim"),
        ("tokenizer_input_shape", "config.tokenizer_input_shape"),
        ("tokenizer_args_in_ckpt", "config.tokenizer_args_in_ckpt"),
        ("tokenizer_unchanged", "result.tokenizer_unchanged"),
        ("tokenizer_digest_before", "result.tokenizer_digest_before"),
        ("tokenizer_digest_after", "result.tokenizer_digest_after"),
        ("tokenizer_frozen", "result.tokenizer_frozen.all_frozen"),
        ("tokenizer_frozen_tensors", "result.tokenizer_frozen.frozen_tensors"),
        ("tokenizer_trainable_tensors", "result.tokenizer_frozen.trainable_tensors"),
        ("adapter_trainable_tensors", "result.adapters.trainable_tensors"),
        ("adapter_frozen_tensors", "result.adapters.frozen_tensors"),
    ):
        b.put(data_model, key, path)
    data_model["sid_shape"] = b.missing_field(
        "config.sid_shape",
        note="run.json records n_items/per-column stats but not the SID array "
             "shape; read it from the .npy without guessing")
    data_model["sid_stats"] = b.value("config.sid_stats")

    # ---- 3. training process --------------------------------------------
    training = {}
    for key, path in (
        ("num_epochs_executed", "result.num_epochs_executed"),
        ("num_epochs_requested", "result.num_epochs_requested"),
        ("best_epoch", "result.best_epoch"),
        ("early_stopped", "result.early_stopped"),
        ("n_best_saves", "result.n_best_saves"),
        ("n_test_evals", "result.n_test_evals"),
        ("elapsed_seconds", "result.elapsed_seconds"),
        ("n_train_sequences", "result.n_train_sequences"),
        ("n_valid_sequences", "result.n_valid_sequences"),
        ("n_test_sequences", "result.n_test_sequences"),
        ("best_ckpt", "result.best_ckpt"),
        ("last_ckpt", "result.last_ckpt"),
    ):
        b.put(training, key, path)
    training["epoch_index_first"] = epochs[0].get("epoch") if epochs else None
    training["epoch_index_last"] = epochs[-1].get("epoch") if epochs else None
    training["n_epoch_records"] = len(epochs)

    # ---- 4. best-epoch validation metrics --------------------------------
    valid = {}
    for key, name in (("valid_recalls", "Recall@5"), ("valid_recalls", "Recall@10"),
                      ("valid_recalls", "Recall@20"), ("valid_ndcgs", "NDCG@5"),
                      ("valid_ndcgs", "NDCG@10"), ("valid_ndcgs", "NDCG@20")):
        valid[name] = (best_record or {}).get(key, {}).get(name)
        if best_record is None:
            b.missing_field(f"result.epochs[best_epoch].{key}.{name}",
                            note="no epoch record matches result.best_epoch")
    valid["source_epoch"] = best_epoch

    # ---- 5. test metrics (recorded only, never used for selection) -------
    test = {}
    for key, name in (("test_recalls", "Recall@5"), ("test_recalls", "Recall@10"),
                      ("test_recalls", "Recall@20"), ("test_ndcgs", "NDCG@5"),
                      ("test_ndcgs", "NDCG@10"), ("test_ndcgs", "NDCG@20")):
        test[name] = (best_record or {}).get(key, {}).get(name)
    test["source_epoch"] = best_epoch
    test["note"] = ("test metrics are recorded at the epoch selected on valid "
                    "NDCG@20; they are not used for model selection")
    test["n_test_evals"] = get(res, "n_test_evals")

    # ---- 6. loss / exception / reproducibility ---------------------------
    losses = {
        "train_loss_at_best_epoch": (best_record or {}).get("train_loss"),
        "train_loss_last_epoch": (epochs[-1].get("train_loss") if epochs else None),
        "stage_loss_contract_flags": get(res, "losses"),
        "stage_loss_contract_note": ("result.losses is the static stage contract "
                                     "(L_SIA/L_PSA true for the recommender "
                                     "stage); the live mounts are "
                                     "result.mounts + config.lambda_*"),
        "mounts": get(res, "mounts"),
        "config_loss_type": get(cfg, "rqvae_config.loss_type"),
        "config_quant_loss_weight": get(cfg, "rqvae_config.quant_loss_weight"),
    }
    for key in ("L_T5", "L_SIA", "L_PSA", "L_RQVAE"):
        if best_record is not None and key not in best_record:
            b.missing_field(f"result.epochs[best_epoch].{key}",
                            note="per-epoch loss components are not recorded; "
                                 "only the aggregate train_loss is")
    losses["per_component_loss_values"] = None

    exceptions = {
        "present_in_run_json": any(
            key in res for key in ("error", "exception", "traceback", "failure")),
        "fields": [k for k in ("error", "exception", "traceback", "failure")
                   if k in res],
        "note": "no exception fields exist in this successful run.json",
    }

    reproducibility = {}
    for key, path in (
        ("device", "config.device"),
        ("num_workers", "config.num_workers"),
        ("beam_size", "config.beam_size"),
        ("infer_size", "config.infer_size"),
        ("topk_list", "config.topk_list"),
        ("limit_train", "config.limit_train"),
        ("limit_eval", "config.limit_eval"),
        ("truncated", "result.truncated"),
        ("allow_unregistered_sid", "config.allow_unregistered_sid"),
        ("embedding_path_explicit", "config.embedding_path_explicit"),
        ("entry", "config.entry"),
        ("protocol", "config.protocol"),
        ("rqvae_config", "config.rqvae_config"),
        ("t5_config", "config.t5_config"),
    ):
        b.put(reproducibility, key, path)

    summary = {
        "meta": {
            "source_run_json": os.path.abspath(source_path),
            "source_sha256": sha256_file(source_path),
            "source_bytes": os.path.getsize(source_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "integration/summarize_l0_run.py",
            "original_preserved": True,
        },
        "experiment_config": experiment,
        "data_and_model": data_model,
        "training": training,
        "best_valid_metrics": valid,
        "test_metrics": test,
        "losses": losses,
        "exceptions": exceptions,
        "reproducibility": reproducibility,
        "field_paths": b.paths,
        "missing_fields": b.missing,
    }
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-json", required=True,
                   help="path to <arm>/run/run.json (read-only)")
    p.add_argument("--out", default=None,
                   help="output path (default: <run dir>/run_summary.json)")
    args = p.parse_args(argv)

    source = args.run_json
    if not os.path.isfile(source):
        print(f"ERROR: not a file: {source}", file=sys.stderr)
        return 2
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(source)),
                                   "run_summary.json")
    if os.path.abspath(out) == os.path.abspath(source):
        print("ERROR: refusing to overwrite the original run.json", file=sys.stderr)
        return 2

    with open(source) as fh:
        doc = json.load(fh)
    summary = build_summary(doc, source)

    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, out)

    print(f"source : {os.path.abspath(source)}")
    print(f"summary: {os.path.abspath(out)}")
    print(f"arm    : {summary['experiment_config']['arm']}")
    print(f"best   : epoch {summary['training']['best_epoch']} "
          f"NDCG@20={summary['best_valid_metrics']['NDCG@20']}")
    missing = summary["missing_fields"]
    print(f"missing: {len(missing)} field(s)")
    for item in missing:
        print(f"  - {item['field']}: {item['reason']}"
              + (f" ({item['note']})" if item.get("note") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
