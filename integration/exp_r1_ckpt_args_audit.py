#!/usr/bin/env python
"""EXP-R1 -- what do the *new* checkpoints' stored ``args`` actually resolve to?

Run:  python -I integration/exp_r1_ckpt_args_audit.py

The gate registered R1 as: the tokenizer checkpoints written by
``tiger_ext/qspace_train.py`` carry an ``args`` namespace that **omits**
``tau``, ``lambda_trans``, ``lambda_balance`` and ``trans_graph_path`` (the
baseline checkpoints written by ``rqvae/main.py`` do carry them), and
``rqvae/trainer.py`` reads each one through ``getattr(args, ..., default)``.
The question is not "does ``getattr`` exist" -- it is whether the value that
is actually in force equals the frozen R13 recipe, and whether anything
downstream is left uncertain by the omission.

The harness therefore resolves values, it does not grep for defaults:

  [1] provenance -- where the checkpoint ``args`` come from, and which keys the
      namespace constructor enumerates
  [2] static extraction -- every optional field the frozen Trainer reads with
      ``getattr`` and every required field it reads directly, plus the
      argparse default of each (current file *and* the freeze commit), so the
      "default" is not taken on faith
  [3] a fresh checkpoint produced by the **current** code -- missing keys,
      present keys, and the runtime values of a real ``RQVAETrainer`` built
      from that checkpoint's args
  [4] the R13 gate artifact (``lc_smoke``, warmup=50) -- same resolution
  [5] the frozen 768d baseline (the lambda-e arms' tokenizer) -- same
  [6] negative controls -- a present value *is* honoured; a present-but-broken
      ``trans_graph_path`` silently disables L_trans (a different failure
      class); deleting a *required* field raises, so the required/optional
      split is enforced by code, not asserted here
  [7] downstream -- ``generate_code``, ``sid_refresh``'s provenance whitelist,
      ``fixed_train`` and every other checkpoint-args reader in the repo
  [8] scope -- production files unchanged, no EXP-L0 driver, verdict recorded

Every training run is one epoch on a 320-row synthetic table.  Nothing here is
evidence about recommendation quality, and EXP-L0 / EXP-B1 are not launched.
"""
import ast
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tiger_ext.qspace_train as QT                                   # noqa: E402
import tiger_ext.sid_refresh as SR                                    # noqa: E402
from tiger_ext.alt_pipeline import RQVAE_CONFIG                       # noqa: E402

RESULTS = []
TABLE = []          # per-field audit rows, also written as JSON

DATASET = "SynR1"
N_ROWS = 320
DIM = 32
FIXTURE_SEED = 11

#: The freeze commit (``freeze clean TIGER baseline``).
FREEZE_COMMIT = "5edfca0"

#: The seven fields ``rqvae/trainer.py`` reads through ``getattr``.  Kept here
#: as the *expected* set; [2] re-derives the set from the source and fails if
#: it disagrees.
OPTIONAL = ("tau", "lambda_trans", "lambda_balance", "trans_graph_path",
            "align_weight", "align_tau", "align_data_path")

#: What R13 requires for **every** tokenizer arm: only ``lambda_csg`` changes
#: between the lambda-c arms, and the lambda-e arms reuse a *frozen* tokenizer.
#: Source: plan §19.1 (frozen Clean TIGER config) + §19.11.3 (arm table).
FROZEN_VALUE = {
    "tau": 0.1, "lambda_trans": 0.0, "lambda_balance": 0.0,
    "trans_graph_path": None,
    "align_weight": 0.0, "align_tau": 0.1, "align_data_path": None,
}

LC_SMOKE = os.path.join(HERE, "artifacts", "r13_gate", "lc_smoke",
                        "Oct-19-2026_00-43-49", "best_collision_model.pth")
BASELINE = os.path.join(REPO, "rqvae", "ckpt", "Toys_Content768d_Baseline",
                        "Jul-31-2026_21-37-33", "best_collision_model.pth")

#: Production files R1 is allowed to change (it must not) -- recorded in [8].
SCOPE = {
    "tiger_ext/qspace_train.py": "305c50a7bf530f53",
    "tiger_ext/fixed_train.py": "389e90854b5f6f13",
    "tiger_ext/sid_refresh.py": "983121cb3d9cd17b",
    "tiger_ext/alt_pipeline.py": "fc1b8652a72c04ce",
}


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    print(f"\n{title}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# source-level extraction
# --------------------------------------------------------------------------- #
def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:                                          # noqa: BLE001
        return "<expr>"


def _receiver_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def parse_getattr_defaults(source):
    """``getattr(<args|self.args>, '<field>', <default>)`` -> field -> info."""
    out = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2):
            continue
        if _receiver_name(node.args[0]) not in ("args", "self.args"):
            continue
        field = node.args[1]
        if not (isinstance(field, ast.Constant) and isinstance(field.value, str)):
            continue
        out.setdefault(field.value, {
            "default": literal(node.args[2]) if len(node.args) > 2 else "<no default>",
            "line": node.lineno,
            "receiver": _receiver_name(node.args[0]),
        })
    return out


def parse_direct_reads(source):
    """``args.<field>`` / ``self.args.<field>`` read sites."""
    out = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            recv = _receiver_name(node.value)
            if recv in ("args", "self.args"):
                out.setdefault(node.attr, node.lineno)
    return out


def parse_argparse_defaults(source):
    """``--flag`` -> default (dash converted to underscore)."""
    out = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)
                and first.value.startswith("--")):
            continue
        name = first.value[2:].replace("-", "_")
        kw = {k.arg: k.value for k in node.keywords}
        out[name] = (literal(kw["default"]) if "default" in kw
                     else "<no default>")
    return out


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def frozen_main_source():
    return subprocess.run(
        ["git", "show", f"{FREEZE_COMMIT}:rqvae/main.py"],
        cwd=REPO, check=True, capture_output=True, text=True).stdout


def runtime_resolution(ckpt_path, tmp_root, tag):
    """Build the **real** frozen Trainer from a checkpoint's stored args."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored = copy.copy(ck["args"])
    args = copy.copy(stored)
    args.ckpt_dir = os.path.join(tmp_root, f"trainer_{tag}")
    args.device = "cpu"
    mods = QT.frozen_tokenizer_modules()
    model = mods["RQVAE"](
        in_dim=DIM, num_emb_list=list(args.num_emb_list), e_dim=args.e_dim,
        layers=list(args.layers), dropout_prob=args.dropout_prob, bn=args.bn,
        loss_type=args.loss_type, quant_loss_weight=args.quant_loss_weight,
        beta=args.beta, kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters, sk_epsilons=list(args.sk_epsilons),
        sk_iters=args.sk_iters)
    trainer = mods["RQVAETrainer"](args, model, 1)
    runtime = {
        "tau": float(trainer.tau),
        "contrastive_loss_tau": float(trainer.contrastive_loss.tau),
        "lambda_trans": float(trainer.lambda_trans),
        "lambda_balance": float(trainer.lambda_balance),
        # trans_graph_path / align_data_path are not attributes; their effect
        # is observable through the resources they would have loaded.
        "trans_graph_path": None if trainer.trans_dict is None else "<loaded>",
        "align_weight": float(trainer.align_weight),
        "align_tau": float(trainer.align_tau),
        "align_data_path": (None if trainer.align_table is None else "<loaded>"),
        "align_proj_is_none": trainer.align_proj is None,
        "trans_dict_is_none": trainer.trans_dict is None,
    }
    return ck, stored, args, trainer, runtime


def audit_checkpoint(path, label, tmp_root):
    """One row per optional field: absent? stored? effective? frozen?"""
    ck, stored, args, trainer, runtime = runtime_resolution(path, tmp_root, label)
    present = {f: hasattr(stored, f) for f in OPTIONAL}
    rows = []
    for field in OPTIONAL:
        effective = runtime[field]
        rows.append({
            "ckpt": label,
            "field": field,
            "stored_key_present": present[field],
            "stored_value": (getattr(stored, field) if present[field] else None),
            "effective_value": effective,
            "frozen_value": FROZEN_VALUE[field],
            "matches_frozen": effective == FROZEN_VALUE[field],
        })
    TABLE.extend(rows)
    return ck, stored, runtime, rows


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #
def fixture_embeddings():
    rng = np.random.RandomState(FIXTURE_SEED)
    return rng.randn(N_ROWS, DIM).astype(np.float32)


def build_fixture(root):
    data_dir = os.path.join(root, DATASET)
    os.makedirs(data_dir, exist_ok=True)
    emb = fixture_embeddings()
    path = os.path.join(data_dir, "item_emb.parquet")
    pd.DataFrame({"embedding": [row for row in emb]}).to_parquet(path)
    return path


def train_fresh_ckpt(root):
    """One 1-epoch run of the **current** qspace_train; returns its ckpt path."""
    cfg = QT.CSGTrainerConfig(
        dataset=DATASET, data_root=root, lambda_csg=0.0, device="cpu",
        seed=2024, epochs=1, eval_step=1, batch_size=N_ROWS, num_workers=0,
        warmup_epochs=0, limit=N_ROWS,
        ckpt_dir=os.path.join(root, "ckpt_fresh"))
    report = QT.train_tokenizer(cfg, hook=QT.CSGHook(lambda_csg=0.0))
    run_dir = report["ckpt_dir"]
    for name in ("best_collision_model.pth", "best_loss_model.pth"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path, report
    raise RuntimeError(f"no checkpoint written under {run_dir}")


# --------------------------------------------------------------------------- #
# [1] provenance of the stored args
# --------------------------------------------------------------------------- #
def part1_provenance(root):
    section("[1] where the checkpoint args come from")
    ns_src = read(os.path.join(REPO, "tiger_ext", "alt_pipeline.py"))
    qt_src = read(os.path.join(REPO, "tiger_ext", "qspace_train.py"))
    tr_src = read(os.path.join(REPO, "rqvae", "trainer.py"))

    check("CSGTrainerConfig.namespace() delegates to rqvae_namespace()",
          "rqvae_namespace(" in qt_src and "def namespace" in qt_src)
    check("rqvae_namespace builds a plain argparse.Namespace",
          "argparse.Namespace(" in ns_src)
    check("the frozen Trainer stores its args verbatim in every checkpoint",
          '"args": self.args' in tr_src)

    cfg = QT.CSGTrainerConfig(dataset=DATASET, data_root=root, device="cpu",
                              ckpt_dir=os.path.join(root, "ckpt_ns"))
    ns = cfg.namespace()
    ns_keys = set(vars(ns).keys())
    check("the namespace constructor does not enumerate any of the 7 fields",
          not (ns_keys & set(OPTIONAL)),
          sorted(ns_keys & set(OPTIONAL)) or "none of 7")
    check("RQVAE_CONFIG does not supply them either",
          not (set(RQVAE_CONFIG) & set(OPTIONAL)),
          sorted(set(RQVAE_CONFIG) & set(OPTIONAL)) or "none of 7")
    return ns_keys


# --------------------------------------------------------------------------- #
# [2] static extraction + default provenance
# --------------------------------------------------------------------------- #
def part2_static(ns_keys, root):
    section("[2] what the Trainer reads, and where each default is written")
    trainer_src = read(os.path.join(REPO, "rqvae", "trainer.py"))
    current_main = parse_argparse_defaults(read(os.path.join(REPO, "rqvae", "main.py")))
    frozen_main = parse_argparse_defaults(frozen_main_source())

    getattrs = parse_getattr_defaults(trainer_src)
    check("the getattr-with-default set is exactly the 7 expected fields",
          set(getattrs) == set(OPTIONAL),
          f"{sorted(getattrs)}")
    for field in OPTIONAL:
        info = getattrs[field]
        print(f"      {field:18s} getattr default={info['default']!r:8} "
              f"(trainer.py:{info['line']})")

    direct = parse_direct_reads(trainer_src)
    required = sorted(set(direct) - set(getattrs))
    check("every directly-read field is present in the new namespace",
          set(required) <= ns_keys,
          f"missing={sorted(set(required) - ns_keys) or 'none'} "
          f"({len(required)} fields)")

    mismatch_frozen = {f: (getattrs[f]["default"], frozen_main.get(f, "<absent>"))
                       for f in OPTIONAL
                       if getattrs[f]["default"] != frozen_main.get(f, "<absent>")}
    check("each getattr default equals the argparse default of the freeze commit",
          not mismatch_frozen,
          mismatch_frozen or f"{FREEZE_COMMIT}:rqvae/main.py, 7/7")

    mismatch_head = {f: (getattrs[f]["default"], current_main.get(f, "<absent>"))
                     for f in OPTIONAL
                     if getattrs[f]["default"] != current_main.get(f, "<absent>")}
    check("each getattr default equals today's argparse default",
          not mismatch_head, mismatch_head or "7/7")

    check("the freeze commit actually declares all 7 flags (comparison not vacuous)",
          all(f in frozen_main for f in OPTIONAL),
          f"{sum(f in frozen_main for f in OPTIONAL)}/7")
    check("frozen values are the ones R13 requires (zeros + tau 0.1)",
          all(frozen_main.get(f, "<absent>") == FROZEN_VALUE[f] for f in OPTIONAL),
          {f: frozen_main.get(f, "<absent>") for f in OPTIONAL})
    return required, getattrs


# --------------------------------------------------------------------------- #
# [3] a checkpoint produced by the current code
# --------------------------------------------------------------------------- #
def part3_fresh(root, ns_keys):
    section("[3] a checkpoint written by the current qspace_train")
    build_fixture(root)
    path, report = train_fresh_ckpt(root)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    stored = ck["args"]
    keys = set(vars(stored).keys())

    check("the fresh checkpoint stores the namespace key set verbatim",
          keys == ns_keys,
          f"extra={sorted(keys - ns_keys)} missing={sorted(ns_keys - keys)}")
    absent = sorted(set(OPTIONAL) - keys)
    check("the fresh checkpoint omits all 7 optional aux-loss fields",
          absent == sorted(OPTIONAL), f"absent={absent}")
    check("the fresh checkpoint carries every required Trainer field",
          set(parse_direct_reads(read(os.path.join(REPO, "rqvae", "trainer.py"))))
          - set(OPTIONAL) <= keys)

    ck, stored, runtime, rows = audit_checkpoint(path, "fresh", root)
    for row in rows:
        print(f"      {row['field']:18s} present={str(row['stored_key_present']):5s} "
              f"effective={row['effective_value']!r:8} frozen={row['frozen_value']!r}")
    check("fresh ckpt: all 7 effective values equal the frozen recipe",
          all(r["matches_frozen"] for r in rows),
          [r["field"] for r in rows if not r["matches_frozen"]] or "7/7")
    check("fresh ckpt: the contrastive module really gets tau=0.1",
          runtime["contrastive_loss_tau"] == FROZEN_VALUE["tau"],
          runtime["contrastive_loss_tau"])
    check("fresh ckpt: no transition graph is loaded",
          runtime["trans_dict_is_none"] is True)
    check("fresh ckpt: no alignment table/projection is built",
          runtime["align_data_path"] is None and runtime["align_proj_is_none"])

    # lambda_csg is the treatment variable; it is *not* in the stored args.
    check("lambda_csg is not stored in the checkpoint args (recorded, see [7])",
          "lambda_csg" not in keys)
    check("the run report does record lambda_csg",
          "lambda_csg" in report,
          report.get("lambda_csg"))
    return path


# --------------------------------------------------------------------------- #
# [4]/[5] the two pre-existing artifact families
# --------------------------------------------------------------------------- #
def part4_existing(root):
    section("[4] R13 gate artifact (lc_smoke, warmup=50)")
    check("the lc_smoke checkpoint exists", os.path.isfile(LC_SMOKE), LC_SMOKE)
    if os.path.isfile(LC_SMOKE):
        ck, stored, runtime, rows = audit_checkpoint(LC_SMOKE, "lc_smoke", root)
        absent = sorted(set(OPTIONAL) - set(vars(stored).keys()))
        check("lc_smoke ckpt omits all 7 optional fields", absent == sorted(OPTIONAL),
              f"absent={absent}")
        check("lc_smoke ckpt: all 7 effective values equal the frozen recipe",
              all(r["matches_frozen"] for r in rows),
              [r["field"] for r in rows if not r["matches_frozen"]] or "7/7")
        check("lc_smoke ckpt records warmup_epochs=50 (B2 fix visible)",
              int(getattr(stored, "warmup_epochs", -1)) == 50,
              getattr(stored, "warmup_epochs", "<absent>"))

    section("[5] frozen 768d baseline checkpoint (the lambda-e arms' tokenizer)")
    check("the baseline checkpoint exists", os.path.isfile(BASELINE), BASELINE)
    if os.path.isfile(BASELINE):
        ck, stored, runtime, rows = audit_checkpoint(BASELINE, "baseline", root)
        keys = set(vars(stored).keys())
        missing = sorted(set(OPTIONAL) - keys)
        check("baseline ckpt carries the 4 Track-B keys",
              {"tau", "lambda_trans", "lambda_balance", "trans_graph_path"} <= keys)
        check("baseline ckpt omits only the 3 E50-b2 align fields",
              missing == ["align_data_path", "align_tau", "align_weight"],
              f"missing={missing}")
        check("baseline ckpt: all 7 effective values equal the frozen recipe",
              all(r["matches_frozen"] for r in rows),
              [r["field"] for r in rows if not r["matches_frozen"]] or "7/7")


# --------------------------------------------------------------------------- #
# [6] negative controls
# --------------------------------------------------------------------------- #
def part6_controls(fresh_ckpt, root):
    section("[6] negative controls -- absent vs present, and required vs optional")
    ck = torch.load(fresh_ckpt, map_location="cpu", weights_only=False)

    _, _, args_a, trainer_a, _ = runtime_resolution(fresh_ckpt, root, "ctl_a")
    args_a.lambda_balance = 0.7
    args_a.tau = 0.07
    mods = QT.frozen_tokenizer_modules()
    model = mods["RQVAE"](
        in_dim=DIM, num_emb_list=list(args_a.num_emb_list), e_dim=args_a.e_dim,
        layers=list(args_a.layers), dropout_prob=args_a.dropout_prob, bn=args_a.bn,
        loss_type=args_a.loss_type, quant_loss_weight=args_a.quant_loss_weight,
        beta=args_a.beta, kmeans_init=args_a.kmeans_init,
        kmeans_iters=args_a.kmeans_iters, sk_epsilons=list(args_a.sk_epsilons),
        sk_iters=args_a.sk_iters)
    trainer_a = mods["RQVAETrainer"](args_a, model, 1)
    check("a present value IS honoured (lambda_balance=0.7 -> 0.7)",
          trainer_a.lambda_balance == 0.7, trainer_a.lambda_balance)
    check("a present tau IS honoured (tau=0.07 -> contrastive tau 0.07)",
          float(trainer_a.contrastive_loss.tau) == 0.07,
          float(trainer_a.contrastive_loss.tau))

    _, _, args_b, _, _ = runtime_resolution(fresh_ckpt, root, "ctl_b")
    args_b.lambda_trans = 0.7
    args_b.trans_graph_path = os.path.join(root, "does_not_exist.npy")
    model_b = mods["RQVAE"](
        in_dim=DIM, num_emb_list=list(args_b.num_emb_list), e_dim=args_b.e_dim,
        layers=list(args_b.layers), dropout_prob=args_b.dropout_prob, bn=args_b.bn,
        loss_type=args_b.loss_type, quant_loss_weight=args_b.quant_loss_weight,
        beta=args_b.beta, kmeans_init=args_b.kmeans_init,
        kmeans_iters=args_b.kmeans_iters, sk_epsilons=list(args_b.sk_epsilons),
        sk_iters=args_b.sk_iters)
    trainer_b = mods["RQVAETrainer"](args_b, model_b, 1)
    check("present-but-broken trans_graph_path silently disables L_trans "
          "(different failure class from an absent key)",
          trainer_b.lambda_trans == 0.0 and trainer_b.trans_dict is None,
          f"lambda_trans={trainer_b.lambda_trans}")

    _, _, args_c, _, _ = runtime_resolution(fresh_ckpt, root, "ctl_c")
    delattr(args_c, "warmup_epochs")
    model_c = mods["RQVAE"](
        in_dim=DIM, num_emb_list=list(args_c.num_emb_list), e_dim=args_c.e_dim,
        layers=list(args_c.layers), dropout_prob=args_c.dropout_prob, bn=args_c.bn,
        loss_type=args_c.loss_type, quant_loss_weight=args_c.quant_loss_weight,
        beta=args_c.beta, kmeans_init=args_c.kmeans_init,
        kmeans_iters=args_c.kmeans_iters, sk_epsilons=list(args_c.sk_epsilons),
        sk_iters=args_c.sk_iters)
    raised = None
    try:
        mods["RQVAETrainer"](args_c, model_c, 1)
    except Exception as exc:                                   # noqa: BLE001
        raised = type(exc).__name__
    check("deleting a *required* field raises (split enforced by code)",
          raised == "AttributeError", raised)

    stored = ck["args"]
    check("absent != present-at-default: hasattr is False on the fresh ckpt",
          not hasattr(stored, "lambda_trans") and
          getattr(stored, "lambda_trans", 0.0) == 0.0)


# --------------------------------------------------------------------------- #
# [7] downstream consumers
# --------------------------------------------------------------------------- #
def part7_downstream(fresh_ckpt):
    section("[7] who else reads checkpoint args, and what they do with the gap")
    gen_src = read(os.path.join(REPO, "rqvae", "generate_code.py"))
    gen_reads = set(parse_direct_reads(gen_src)) - set(parse_getattr_defaults(gen_src))
    ck = torch.load(fresh_ckpt, map_location="cpu", weights_only=False)
    keys = set(vars(ck["args"]).keys())

    check("generate_code reads no optional aux-loss field",
          not (gen_reads & set(OPTIONAL)),
          sorted(gen_reads & set(OPTIONAL)) or "none")
    check("generate_code's fields are all present in the new ckpt",
          gen_reads <= keys,
          f"missing={sorted(gen_reads - keys) or 'none'}")

    check("sid_refresh's provenance whitelist omits all 7 (gap, not error)",
          not (set(SR._CKPT_ARG_KEYS) & set(OPTIONAL)),
          sorted(set(SR._CKPT_ARG_KEYS) & set(OPTIONAL)) or "none of 7")
    prov = SR.ckpt_args(fresh_ckpt)
    check("sid_refresh provenance still records the model identity",
          {"data_path", "num_emb_list", "e_dim", "layers", "seed"} <= set(prov),
          sorted(prov))

    fixed_src = read(os.path.join(REPO, "tiger_ext", "fixed_train.py"))
    check("fixed_train never reads a checkpoint's `args` at all",
          not re.search(r"\[\s*['\"]args['\"]\s*\]", fixed_src))

    # every file that loads checkpoint args, does it read one of the 7?
    readers = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "integration", "__pycache__", "doc")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                src = read(path)
            except (UnicodeDecodeError, OSError):
                continue
            if not re.search(r"\[\s*['\"]args['\"]\s*\]|get\(\s*['\"]args['\"]", src):
                continue
            used = set(parse_direct_reads(src)) & set(OPTIONAL)
            readers.append((os.path.relpath(path, REPO), sorted(used)))
    offenders = [(p, u) for p, u in readers if u]
    check("no checkpoint-args reader in the repo touches the 7 optional fields",
          not offenders, offenders or f"{len(readers)} readers checked")

    # the treatment variable has the same auditability gap
    check("lambda_csg is likewise not a checkpoint field (report-only)",
          "lambda_csg" not in keys)


# --------------------------------------------------------------------------- #
# [8] scope + verdict
# --------------------------------------------------------------------------- #
def part8_scope(root):
    section("[8] scope and verdict")
    mismatched = {}
    for rel, prefix in SCOPE.items():
        digest = sha256_file(os.path.join(REPO, rel))[:16]
        if digest != prefix:
            mismatched[rel] = digest
    check("every production file is byte-identical to its recorded hash",
          not mismatched, mismatched or f"{len(SCOPE)}/{len(SCOPE)}")

    hits = subprocess.run(
        ["grep", "-rn", "EXP_L0", "tools", "rqvae", "tiger_ext"],
        cwd=REPO, capture_output=True, text=True)
    check("no EXP-L0 driver exists (the gate is still closed)",
          hits.returncode != 0 and not hits.stdout.strip(),
          hits.stdout.strip()[:200] or "no hits")

    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    expected = "\n".join([
        "M model/main.py", " M rqvae/models/rq.py", " M rqvae/models/vq.py",
        " M rqvae/trainer.py", "?? integration/", "?? tiger_ext/"])
    check("the working tree is exactly the pre-task snapshot",
          status == expected,
          "unchanged" if status == expected else status.replace("\n", " | "))

    all_frozen = all(row["matches_frozen"] for row in TABLE)
    check("every audited (checkpoint, field) pair resolves to the frozen value",
          all_frozen,
          f"{sum(r['matches_frozen'] for r in TABLE)}/{len(TABLE)}")
    return all_frozen


# --------------------------------------------------------------------------- #
def main():
    root = tempfile.mkdtemp(prefix="exp_r1_")
    try:
        ns_keys = part1_provenance(root)
        required, _getattrs = part2_static(ns_keys, root)
        fresh = part3_fresh(root, ns_keys)
        part4_existing(root)
        part6_controls(fresh, root)
        part7_downstream(fresh)
        frozen_ok = part8_scope(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    out_dir = os.path.join(HERE, "artifacts", "r1_ckpt_args")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp_r1_audit.json"), "w") as fh:
        json.dump({"table": TABLE, "frozen_value": FROZEN_VALUE,
                   "optional_fields": list(OPTIONAL),
                   "required_fields": required,
                   "freeze_commit": FREEZE_COMMIT}, fh, indent=2)

    passed = sum(1 for _name, ok, _detail in RESULTS if ok)
    total = len(RESULTS)
    verdict = "CLOSED-BY-AUDIT" if (frozen_ok and passed == total) else "BLOCKED"
    print("\n" + "=" * 72)
    print(f"EXP-R1: {passed}/{total} checks passed")
    print(f"R1 verdict: {verdict}  (audit table -> {out_dir}/exp_r1_audit.json)")
    print("=" * 72)
    if passed != total:
        print("\nFailures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name} -- {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
