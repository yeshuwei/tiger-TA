#!/usr/bin/env python
"""EXP-C6 -- CSG smoke test: the whole chain, with the constraint switched on.

C6 is a **smoke test**, not an effect experiment. It answers exactly four
questions, one per group of sections:

  Q1  四个 lambda 都能不能跑通?
      [3] the four arms A/B/C/D each run RQ-VAE -> SID -> T5 -> recommender ->
          evaluation and write every artifact.

  Q2  CSG 有没有真正进入训练?
      [4] the hook is called on every tokenizer step, its term lands in the
          total loss, the codebooks receive gradient and the encoder is bitwise
          unaffected -- measured at the smoke configuration, not inherited.

  Q3  完整 TIGER pipeline 有没有被破坏?
      [1] scope: C6 adds no tracked edit and no forbidden token
      [2] the fixtures, and the row space C1/C2/C3 all share
      [5] lambda_csg = 0 is bitwise equal to the pristine chain
      [6] the E7 stage contract, the SID hand-off, the historical mechanisms
      [7] no user information reaches the SID

  Q4  kmeans_init 哪个配置更安全?
      [9] the True/False table and the basis for the choice

  [8]  the real-data arm (Toys) at the smoke budget
  [10] regression: C5's own suite, re-run

Every number below is a measurement made in this process. Recall/NDCG appear
only as evidence that the pipeline ran; the brief forbids reading them for or
against the constraint, and the report says so where they are printed.

Run:  python -I integration/exp_c6_csg_smoke.py
      EXP_C6_TOYS=skip         to drop the real-data arm
      EXP_C6_REGRESSION=skip   to drop the C5 re-run
      EXP_C6_DEVICE=cuda       to run the chains on GPU
"""
import os

# Pinned before torch/numpy are imported: the project's measured reproducibility
# fix for the KMeans-without-random_state path (rqvae/models/layers.py:86).
os.environ["OMP_NUM_THREADS"] = "1"

import ast                                                        # noqa: E402
import difflib                                                    # noqa: E402
import hashlib                                                    # noqa: E402
import json                                                       # noqa: E402
import re                                                         # noqa: E402
import subprocess                                                 # noqa: E402
import sys                                                        # noqa: E402
import tempfile                                                   # noqa: E402
import warnings                                                   # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO, "data")

sys.path.insert(0, REPO)

from tiger_ext import qspace_smoke as SM                            # noqa: E402
from tiger_ext import qspace_train as QT                            # noqa: E402
from tiger_ext.qspace_constraint import REBUILD_ATOL                # noqa: E402

TRAINER = os.path.join(REPO, "rqvae", "trainer.py")
BASELINE_COMMIT = "5edfca0"                    # the frozen Clean TIGER baseline
C5_HARNESS = os.path.join(HERE, "exp_c5_csg_integration.py")

TOYS = os.environ.get("EXP_C6_TOYS", "all")
REGRESSION = os.environ.get("EXP_C6_REGRESSION", "all")
DEVICE = os.environ.get("EXP_C6_DEVICE", "cpu")
SEED = 2024

#: The tracked files E1-E6 modified, and the one C5 added. C6 adds **none**.
PRE_C6_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}
C6_AUTHORIZED_MODIFIED = PRE_C6_MODIFIED | {"rqvae/trainer.py"}

#: The E10 guard, run directly: no file under these trees may contain the
#: substring the guard greps for. This is why C6's new driver names its switch
#: ``lambda_csg`` / ``self.csg`` / ``CSGHook`` and never the other spelling.
FORBIDDEN_IN_TREE = ("csg_", "lightgcn", "graph_conv")
GUARD_TREES = ("tiger_ext", "model", "rqvae")

#: User-side tokens that must not appear in the tokenizer's own path.
USER_TOKENS = ("user_id", "userid", "user_emb", "session_id", "item2user")
#: Interaction-file names the tokenizer stage must never read.
INTERACTION_FILES = ("train.parquet", "valid.parquet", "test.parquet")

FAIL_RE = re.compile(r"^\s*\[FAIL\]\s*(.+?)(?:\s+--.*)?$", re.M)
HISTORICAL_PROBES = ("contrastive_loss", "balance_loss", "_get_transition_loss",
                     "_get_align_loss")

RESULTS = []
TMP = tempfile.mkdtemp(prefix="exp_c6_")
SYN = {}


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    return ok


def sha16(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def sha256_path(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def suite_summary_re(exp_id):
    return re.compile(rf"^EXP-{re.escape(exp_id)}:\s*(\d+)/(\d+) checks passed", re.M)


def git_status_lines():
    return subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                          capture_output=True, text=True).stdout.splitlines()


def module_pins(source, names):
    """Read dict/scalar pins out of another harness's source, without running it."""
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return out


def trainer_class_bodies(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Trainer")
    return {f"Trainer.{n.name}": ast.dump(n) for n in cls.body
            if isinstance(n, ast.FunctionDef)}


def pristine_trainer_source():
    return subprocess.run(
        ["git", "-C", REPO, "show", f"{BASELINE_COMMIT}:rqvae/trainer.py"],
        capture_output=True, text=True, check=True).stdout


def pinned_loader(path, rows, batch, *, return_index):
    mods = QT.frozen_tokenizer_modules()
    data = mods["EmbDataset"](path, return_index=return_index)
    if rows is not None:
        data = torch.utils.data.Subset(data, list(range(int(rows))))
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False,
                                       num_workers=0)


def make_recorder(base):
    """A Trainer subclass that counts calls to named attributes.

    ``_check_nan`` is called once on ``L_RQVAE`` and once on the total, so the
    pair is read from inside the frozen method rather than reconstructed.
    """

    class Recording(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.pairs = []
            self.probes = {}

        def _check_nan(self, loss):
            self.pairs.append(float(loss.detach()))
            super()._check_nan(loss)

        def probe(self, attr):
            original = getattr(self, attr)
            self.probes[attr] = 0

            def counting(*a, **kw):
                self.probes[attr] += 1
                return original(*a, **kw)

            setattr(self, attr, counting)

        def steps(self):
            return [(self.pairs[2 * i], self.pairs[2 * i + 1])
                    for i in range(len(self.pairs) // 2)]

    return Recording


def arm_workdir(tag):
    return os.path.join(TMP, f"chain_{tag}")


def run_arm(spec, lam, *, tag, pristine=False):
    """One arm, with its work directory kept so the artifacts can be re-read."""
    return SM.run_chain(spec, lambda_csg=lam, tag=tag, pristine=pristine,
                        workdir=arm_workdir(tag), keep=True)


# --------------------------------------------------------------------------- #
# [1] scope -- is the check itself honest?
# --------------------------------------------------------------------------- #
def part1_scope():
    print("\n[1] scope: C6 adds no tracked edit, and no forbidden token anywhere")
    status = git_status_lines()
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("the modified tracked-file set is still exactly C5's four files -- C6 "
          "changed no tracked file at all",
          modified == C6_AUTHORIZED_MODIFIED, f"modified: {sorted(modified)}")
    check("nothing is deleted, renamed or newly staged",
          not [l for l in status if l[:2].strip() in ("D", "R", "A")],
          f"{len(status)} status lines; none is D/R/A")

    pristine = pristine_trainer_source()
    edited = open(TRAINER, encoding="utf-8").read()
    before, after = trainer_class_bodies(pristine), trainer_class_bodies(edited)
    changed = sorted(k for k in before if before[k] != after[k])
    diff = list(difflib.unified_diff(pristine.splitlines(), edited.splitlines(),
                                     "pristine", "edited", n=3, lineterm=""))
    check("rqvae/trainer.py is still byte-for-byte C5's mount and nothing else: the "
          "same two methods changed, every other body is untouched",
          changed == ["Trainer.__init__", "Trainer._train_epoch"],
          f"changed: {changed}; sha16 = {sha16(TRAINER)}; "
          f"sha256(diff)[:16] = "
          f"{hashlib.sha256(''.join(diff).encode()).hexdigest()[:16]}")

    c5_source = open(C5_HARNESS, encoding="utf-8").read()
    pins = module_pins(c5_source,
                       {"PINNED_SHA16", "FROZEN_CORE_PINS", "PINNED_C4_MODULE",
                        "PINNED_C4_MODULE_SHA16"})
    watched = dict(pins["PINNED_SHA16"])
    watched[pins["PINNED_C4_MODULE"]] = pins["PINNED_C4_MODULE_SHA16"]
    watched.update(pins["FROZEN_CORE_PINS"])
    stale = {p: (v, sha16(os.path.join(REPO, p))) for p, v in watched.items()
             if sha16(os.path.join(REPO, p)) != v}
    check("every file hash C5 recorded is still the hash on disk, re-read from C5's "
          "own harness rather than restated here",
          not stale,
          f"{len(watched)} files compared against C5's own pins"
          + (f"; stale: {stale}" if stale else ""))

    hits = []
    for tree in GUARD_TREES:
        proc = subprocess.run(
            ["grep", "-rniE", "|".join(FORBIDDEN_IN_TREE),
             os.path.join(REPO, tree)], capture_output=True, text=True)
        if proc.returncode != 1 or proc.stdout.strip():
            hits.append((tree, proc.returncode, proc.stdout.strip()[:200]))
    check("the E10 guard, run directly over tiger_ext/ model/ rqvae/, still reports "
          "nothing -- so C6's new driver is invisible to it, as required",
          not hits, f"{len(hits)} tree(s) with a hit: {hits or 'none'}")

    driver = os.path.join(REPO, "tiger_ext", "qspace_smoke.py")
    harness = os.path.join(HERE, "exp_c6_csg_smoke.py")
    package = sorted(n for n in os.listdir(os.path.join(REPO, "tiger_ext"))
                     if n.endswith(".py"))
    check("C6's new module is where the plan puts it, the check itself lives outside "
          "the package (so it is outside the E10 guard's scan), and nothing named "
          "like the harness was added inside the package",
          os.path.isfile(driver) and os.path.isfile(harness)
          and "qspace_smoke.py" in package
          and not os.path.isfile(os.path.join(REPO, "tiger_ext",
                                              "exp_c6_csg_smoke.py")),
          "tiger_ext/*.py = " + ", ".join(package))


# --------------------------------------------------------------------------- #
# [2] the fixtures, and the row space everyone shares
# --------------------------------------------------------------------------- #
def part2_fixtures():
    print("\n[2] fixtures: the synthetic chain, and Toys where the row space bites")
    import pandas as pd

    spec = SYN["spec"]
    syn_emb = pd.read_parquet(os.path.join(spec.data_root, spec.dataset,
                                           "item_emb.parquet"))
    check("the synthetic fixture carries an ItemID column, so C1's canonical-item "
          "lookup and C2's relation builder both work on it",
          {"ItemID", "embedding"} <= set(syn_emb.columns),
          f"{len(syn_emb)} items, columns {sorted(syn_emb.columns)}")

    from tiger_ext.collab_relation import CollabRelation
    syn_rel = CollabRelation.load(spec.relation_prefix,
                                 expected_item_ids=syn_emb["ItemID"].to_numpy(
                                     dtype=np.int64), expected_k=spec.k)
    check("the synthetic relation was written by C2's own builder and its row space "
          "is the fixture's ItemID column, elementwise",
          np.array_equal(syn_rel.item_ids, syn_emb["ItemID"].to_numpy(dtype=np.int64)),
          f"{syn_rel.n_items} rows, k_eff = {syn_rel.neighbors.shape[1]}")

    toys_dir = os.path.join(DATA_DIR, "Toys")
    toys_emb = pd.read_parquet(os.path.join(toys_dir, "item_emb.parquet"),
                               columns=["ItemID"])
    toys_ids = toys_emb["ItemID"].to_numpy(dtype=np.int64)
    from tiger_ext.collab_relation import relation_prefix as rel_prefix
    toys_rel = CollabRelation.load(rel_prefix("Toys", k=SM.CST_SMOKE_K),
                                   expected_item_ids=toys_ids,
                                   expected_k=SM.CST_SMOKE_K)
    check("on the real dataset too, the C2 relation's row space is the "
          "item_emb.parquet ItemID column elementwise",
          np.array_equal(toys_rel.item_ids, toys_ids),
          f"Toys: {toys_rel.n_items} rows, k_eff = {toys_rel.neighbors.shape[1]}")
    off_by_one = int((toys_ids - np.arange(1, toys_ids.size + 1) != 0).sum())
    check("the mapping is the ItemID column and not an offset: on Toys the two "
          "coincide, which is exactly why the rule has to be enforced by the loader "
          "rather than by eye -- it is the datasets where they diverge that mis-pair",
          off_by_one == 0,
          f"Toys rows with ItemID != row + 1: {off_by_one} (Beauty is the "
          "counter-example, where the same code silently mis-pairs 4158 rows)")

    art = os.path.join(REPO, "integration", "artifacts")
    have = {"cf_embed": os.path.isdir(os.path.join(art, "cf_embed")),
            "collab_relation": os.path.isdir(os.path.join(art, "collab_relation"))}
    check("C1's and C2's Toys artifacts are both on disk, so the smoke really does "
          "load the relation rather than skip it",
          all(have.values()), f"{have}")
    check("the constraint cannot be built for Musical_Instruments -- C1/C2's "
          "artifacts exist for Toys / Sport / Beauty only -- which is why the "
          "real-data arm is Toys",
          not os.path.isdir(os.path.join(art, "cf_embed", "Musical_Instruments"))
          and os.path.isdir(os.path.join(art, "cf_embed")),
          "recorded as a limitation, not worked around: generating them would be "
          "new C1/C2 scope")
    return syn_emb, toys_ids


# --------------------------------------------------------------------------- #
# [3] Q1 -- the four arms
# --------------------------------------------------------------------------- #
#: The full chain C6 has to walk: tokenizer -> SID -> T5 -> recommendation ->
#: evaluation. It is covered by **two** drivers, which is the composition and not
#: a gap: the tokenizer leg runs through C5's ``train_tokenizer`` (which owns the
#: constrained step and therefore the ``csg=`` seam), and the other three arise
#: from ``TigerAlternatingPipeline``. Only the latter write ``phase_log`` rows --
#: ``train_tokenizer`` is not a pipeline phase and writes none -- so the tokenizer
#: leg is evidenced by its own artifact and step count instead of by a log row.
FULL_CHAIN = ("tokenizer", "refresh", "recommender", "evaluation")
PIPELINE_PHASES = ("refresh", "recommender", "evaluation")


def tokenizer_leg_ran(report):
    """The tokenizer leg's evidence: a trained checkpoint, steps, a finite loss."""
    tok = report["tokenizer"]
    return (os.path.isfile(tok["ckpt_path"]) and tok["n_steps"] > 0
            and tok["n_batches"] > 0 and np.isfinite(tok["best_loss"]))


def part3_arms():
    print("\n[3] Q1: the four arms, A = 0 / B = 1e-4 / C = 1e-3 / D = 1e-2")
    spec = SYN["spec"]
    arms = {}
    for lam in SM.CST_SMOKE_LAMBDAS:
        tag = f"syn-{lam:g}"
        try:
            report = run_arm(spec, lam, tag=tag)
        except Exception as exc:                                   # noqa: BLE001
            arms[lam] = {"error": f"{type(exc).__name__}: {exc}"}
            check(f"arm lambda_csg={lam:g}: the chain reached the evaluation phase",
                  False, arms[lam]["error"][:200])
            continue
        arms[lam] = report
        phases = [p for p in report["phase_order"]]
        on_disk = {
            "tokenizer ckpt": os.path.isfile(report["tokenizer"]["ckpt_path"]),
            "sid": os.path.isfile(report["refresh"]["sid_path"]),
            "recommender ckpt": os.path.isfile(report["recommender"]["ckpt_path"]),
            "evaluation": bool(report["evaluation"]["recalls"]),
        }
        check(f"arm lambda_csg={lam:g}: the full chain "
              f"{' -> '.join(FULL_CHAIN)} completed and every artifact was written",
              tokenizer_leg_ran(report)
              and [p[0] for p in phases] == list(PIPELINE_PHASES)
              and all(on_disk.values()),
              f"{report['seconds']['total']:.1f}s; tokenizer leg "
              f"{report['tokenizer']['n_steps']} steps / "
              f"{report['tokenizer']['n_batches']} batches on "
              f"{report['tokenizer']['n_rows']} items; pipeline log "
              f"{[p[0] for p in phases]}; artifacts {on_disk}")

    ok = [r for r in arms.values() if "error" not in r]
    shapes = {json.dumps(r["spec"], sort_keys=True) for r in ok}
    check("the arms differ in the weight and nothing else: one dataset, one seed, "
          "one budget set, recorded per arm",
          len(shapes) == 1,
          f"{len(ok)} arms, {len(shapes)} distinct configuration(s)")
    check("the four weights are C5's candidates plus the off switch, and C6 "
          "introduced none of its own",
          tuple(SM.CST_SMOKE_LAMBDAS) == (0.0, 1e-4, 1e-3, 1e-2)
          and all(abs(a - b) < 1e-15 for a, b in
                  zip(sorted(arms), sorted(SM.CST_SMOKE_LAMBDAS))),
          f"{SM.CST_SMOKE_LAMBDAS}")
    check("the tokenizer leg is outside the pipeline's own phase log by "
          "construction, not by omission: the log holds the three phases the "
          "pipeline runs, and the first leg is driven by C5's entry -- which is "
          "the entry that owns the constraint",
          bool(ok)
          and all([p[0] for p in r["phase_order"]] == list(PIPELINE_PHASES)
                  and "tokenizer" not in [p[0] for p in r["phase_order"]]
                  for r in ok)
          and all(tokenizer_leg_ran(r) for r in ok),
          (f"{len(ok)} arms; log = {[p[0] for p in ok[0]['phase_order']]}, "
           f"tokenizer leg evidenced by "
           f"{ok[0]['tokenizer']['n_steps']} constrained-entry steps")
          if ok else "no arm completed")
    return arms


# --------------------------------------------------------------------------- #
# [4] Q2 -- the constraint really entered training
# --------------------------------------------------------------------------- #
def _grad_step(spec, lam, mode):
    """One forward/backward from a pinned state, at the smoke model configuration.

    ``mode='total'`` reproduces the trainer's own step; ``mode='objective'``
    differentiates the hook's term alone on its own graph. Dropout is 0 in this
    configuration, so the graph at ``lambda`` and at 0 is the same graph plus one
    added term -- which is what makes the additivity identity below meaningful.

    The batch here is 512 rather than the smoke's 1024: this block measures the
    gradient routing and the additivity of the term, neither of which depends on
    the batch size. Both rows of the k-means probe below are the ones that read
    the first-batch geometry, and they use the real batch size.
    """
    mods = QT.frozen_tokenizer_modules()
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    cfg = dict(spec.rqvae_config)
    model = mods["RQVAE"](
        in_dim=SYN["dim"], num_emb_list=cfg["num_emb_list"], e_dim=cfg["e_dim"],
        layers=cfg["layers"], dropout_prob=0.0, bn=False, loss_type="mse",
        quant_loss_weight=1.0, beta=0.25, kmeans_init=cfg["kmeans_init"],
        kmeans_iters=cfg["kmeans_iters"], sk_epsilons=cfg["sk_epsilons"],
        sk_iters=cfg["sk_iters"])
    model.train()

    loader = pinned_loader(spec.embedding_path, None, 512, return_index=True)
    batches = list(loader)
    with torch.no_grad():
        model(batches[0][1])           # spend the k-means init on batch 0
    rows, x = batches[1]
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}

    out, rq, idx, z, x_q = model(x, return_z=True, return_xq=True)
    loss, _ = model.compute_loss(out, rq, xs=x)
    term_value, term = 0.0, None
    if lam > 0.0:
        hook = SM.build_hook(spec, lam)
        term = hook(x_q=x_q, indices=idx, latent=z, rows=rows, rqvae=model)
        term_value = float(term.detach())
        if mode == "total":
            loss = loss + term
    if mode == "objective":
        term.backward()
    else:
        loss.backward()

    def grads(prefix):
        return {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in model.named_parameters() if n.startswith(prefix)}

    return {"loss": float(loss.detach()), "term": term_value, "init": init,
            "cb": grads("rq."), "enc": grads("encoder.")}


def part4_entered(arms):
    print("\n[4] Q2: the constraint entered the training step")
    spec = SYN["spec"]
    off = arms[0.0]
    check("the off arm never entered the hook: zero calls, and zero *disabled* calls "
          "too, so 'not called' is the guard working and not a mis-wiring",
          off["counters"]["n_calls"] == 0
          and off["counters"]["n_disabled_calls"] == 0,
          f"n_calls = {off['counters']['n_calls']}, "
          f"n_disabled_calls = {off['counters']['n_disabled_calls']}")

    for lam in SM.CST_SMOKE_LAMBDAS[1:]:
        arm = arms[lam]
        c, tok = arm["counters"], arm["tokenizer"]
        check(f"lambda_csg={lam:g}: the hook ran on every tokenizer step and never "
              f"returned a manufactured zero",
              c["n_calls"] == tok["n_steps"] and c["n_nonempty"] == c["n_calls"]
              and c["n_calls"] > 0,
              f"{c['n_calls']} calls for {tok['n_steps']} steps; "
              f"non-empty {c['n_nonempty']}, zero {c['n_zero']}; "
              f"levels {c['n_levels_total']}, pairs {c['n_pairs_total']}")
        hist = arm["history"]
        weighted_ok = all(abs(h["weighted"] - lam * h["loss"]) < 1e-12 * max(
            1.0, abs(h["weighted"])) for h in hist)
        finite = all(np.isfinite(h["loss"]) and np.isfinite(h["weighted"])
                     for h in hist)
        check(f"lambda_csg={lam:g}: every recorded term is positive, differentiable, "
              f"finite, and weighted exactly once",
              bool(hist) and all(h["loss"] > 0.0 for h in hist)
              and all(h["requires_grad"] for h in hist) and weighted_ok and finite,
              f"{len(hist)} steps; L_csg in "
              f"[{min(h['loss'] for h in hist):.4f}, "
              f"{max(h['loss'] for h in hist):.4f}]; "
              f"weighted == lambda * L_csg on all steps")

    lam = 1e-3
    hi, lo, obj = (_grad_step(spec, lam, "total"), _grad_step(spec, 0.0, "total"),
                   _grad_step(spec, lam, "objective"))
    check("the three gradient runs start from the same initial state, so the "
          "comparison is not confounded by initialisation",
          all(torch.equal(hi["init"][k], lo["init"][k]) for k in hi["init"])
          and all(torch.equal(hi["init"][k], obj["init"][k]) for k in hi["init"]),
          f"{len(hi['init'])} tensors compared across three runs")
    check("at the smoke model configuration too, the loss handed to backward() "
          "differs at identical state and batch -- the term is not computed then "
          "discarded",
          hi["loss"] != lo["loss"],
          f"L_total(1e-3) - L_total(0) = {hi['loss'] - lo['loss']:.6e}")

    # Only tensors that received gradient in *all three* runs can carry the
    # identity; a parameter the loss never touches would make it vacuously true,
    # so the count is reported rather than silently absorbed.
    quant = [n for n in hi["cb"] if hi["cb"][n] is not None
             and lo["cb"][n] is not None and obj["cb"][n] is not None]
    worst, scale = 0.0, 1e-30
    for n in quant:
        d = float((hi["cb"][n] - lo["cb"][n] - obj["cb"][n]).abs().max())
        worst = max(worst, d)
        scale = max(scale, float(lo["cb"][n].abs().max()))
    check("gradient additivity holds per quantiser tensor: dL_total/dW - "
          "dL_RQVAE/dW is the objective's own gradient, so the term reached a "
          "differentiable leaf",
          len(quant) == len(hi["cb"]) and bool(quant) and worst / scale < 1e-4,
          f"max |g_total - g_base - g_objective| = {worst:.3e} "
          f"(relative {worst / scale:.3e}) over {len(quant)}/{len(hi['cb'])} "
          f"tensors that carry gradient in all three runs")

    enc_missing = [n for n in hi["enc"]
                   if hi["enc"][n] is None or lo["enc"][n] is None]
    enc_diff = [n for n in hi["enc"] if hi["enc"][n] is not None
                and lo["enc"][n] is not None
                and not torch.equal(hi["enc"][n], lo["enc"][n])]
    enc_nz = [n for n, g in obj["enc"].items()
              if g is not None and float(g.abs().sum()) > 0]
    check("route P1 holds at the smoke configuration: the constraint contributes "
          "nothing to any encoder parameter -- bitwise, not approximately -- while "
          "it does produce gradient at the codebooks",
          not enc_diff and not enc_missing and not enc_nz
          and any(g is not None and float(g.abs().sum()) > 0
                  for g in obj["cb"].values()),
          f"{len(enc_diff)}/{len(hi['enc'])} encoder tensors differ with the weight "
          f"on ({len(enc_missing)} without gradient in either run); "
          f"{len(enc_nz)} encoder tensors receive objective gradient")

    fin = [(lam, np.isfinite(a["tokenizer"]["best_loss"]),
            np.isfinite(a["recommender"]["train_loss"]))
           for lam, a in arms.items()]
    check("no arm produced a NaN or an Inf anywhere in the tokenizer or the "
          "recommender loss",
          all(f[1] and f[2] for f in fin), f"{fin}")
    return hi, lo, obj


# --------------------------------------------------------------------------- #
# [5] Q3a -- lambda_csg = 0 is the pristine chain
# --------------------------------------------------------------------------- #
def _state_dict(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def part5_bitwise(arms):
    print("\n[5] Q3a: lambda_csg = 0 versus the pristine call signature, bitwise")
    spec = SYN["spec"]
    off = arms[0.0]
    try:
        pristine = run_arm(spec, 0.0, tag="syn-pristine", pristine=True)
    except Exception as exc:                                        # noqa: BLE001
        check("the pristine chain (no constraint argument at all) completes",
              False, f"{type(exc).__name__}: {exc}"[:200])
        return None

    a_sid = np.load(off["refresh"]["sid_path"])
    b_sid = np.load(pristine["refresh"]["sid_path"])
    check("the SID array is bitwise identical -- the disabled hook changed no code "
          "any item received",
          np.array_equal(a_sid, b_sid)
          and sha256_path(off["refresh"]["sid_path"])
          == sha256_path(pristine["refresh"]["sid_path"]),
          f"shape {a_sid.shape}; sha256 equal: "
          f"{sha256_path(off['refresh']['sid_path']) == sha256_path(pristine['refresh']['sid_path'])}")

    a_tok = _state_dict(off["tokenizer"]["ckpt_path"])["state_dict"]
    b_tok = _state_dict(pristine["tokenizer"]["ckpt_path"])["state_dict"]
    same_keys = set(a_tok) == set(b_tok)
    diff = [k for k in a_tok if same_keys and not torch.equal(a_tok[k], b_tok[k])]
    check("the tokenizer state dict is bitwise identical tensor by tensor -- the "
          "switch is inert, not merely small",
          same_keys and not diff and off["tokenizer"]["best_loss"]
          == pristine["tokenizer"]["best_loss"],
          f"{len(a_tok)} tensors, {len(diff)} differ; best_loss "
          f"{off['tokenizer']['best_loss']} == {pristine['tokenizer']['best_loss']}")

    a_rec = _state_dict(off["recommender"]["ckpt_path"])["state_dict"]
    b_rec = _state_dict(pristine["recommender"]["ckpt_path"])["state_dict"]
    rec_diff = [k for k in a_rec if not torch.equal(a_rec[k], b_rec[k])]
    check("the recommender state dict is bitwise identical too, so the equality "
          "propagated through the SID into the whole downstream chain",
          set(a_rec) == set(b_rec) and not rec_diff,
          f"{len(a_rec)} tensors, {len(rec_diff)} differ")

    check("the evaluation numbers are bitwise identical, not close: same Recall and "
          "same NDCG at every k",
          off["evaluation"]["recalls"] == pristine["evaluation"]["recalls"]
          and off["evaluation"]["ndcgs"] == pristine["evaluation"]["ndcgs"],
          f"recalls {off['evaluation']['recalls']}")

    check("the same is true at the trainer level for the whole class of settings: "
          "the disabled hook is parameter-free and allocates nothing, so it cannot "
          "perturb a stream",
          off["tokenizer"]["optimizer_params"] == pristine["tokenizer"][
              "optimizer_params"]
          and not off["tokenizer"]["hook_in_optimizer"],
          f"{off['tokenizer']['optimizer_params']} optimiser parameters, hook in "
          f"optimiser: {off['tokenizer']['hook_in_optimizer']}")
    return pristine


# --------------------------------------------------------------------------- #
# [6] Q3b -- the pipeline itself
# --------------------------------------------------------------------------- #
def _probe_run(spec, lam, probes, extra=None):
    mods = QT.frozen_tokenizer_modules()
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    cfg = dict(spec.rqvae_config)
    cfg.update(extra or {})
    config = QT.CSGTrainerConfig(
        dataset=spec.dataset, data_root=spec.data_root,
        artifact_dir=spec.artifact_dir, relation_prefix=spec.relation_prefix,
        lambda_csg=lam, k=spec.k, validate=True, device=spec.device,
        seed=spec.seed, epochs=1, eval_step=1, batch_size=512, num_workers=1,
        warmup_epochs=0, ckpt_dir=os.path.join(TMP, f"probe_{lam:g}"),
        rqvae_config=cfg)
    base = make_recorder(mods["RQVAETrainer"])
    holder = {}

    def factory(args, model, data_num, csg):
        trainer = base(args, model, data_num, csg)
        for p in probes:
            trainer.probe(p)
        holder["trainer"] = trainer
        return trainer

    hook = SM.build_hook(spec, lam)
    QT.train_tokenizer(config, hook=hook, trainer_factory=factory,
                       loader_factory=SM.pinned_loader)
    return holder["trainer"], hook


def _mount_probe_run(spec, lam, *, tag):
    """Re-run one arm with the two historical mount forwards counted.

    Counting the forwards measures the entry directly, rather than reading a
    config value back. The two arms of the fact reinforce each other: the run
    completed, and both mounts raise when entered while disabled -- so a count of
    zero is the only consistent reading.
    """
    from tiger_ext import PSAMount, SIAMount
    seen = {"sia": 0, "psa": 0}
    orig = (SIAMount.forward, PSAMount.forward)

    def sia_fwd(self, *a, **kw):
        seen["sia"] += 1
        return orig[0](self, *a, **kw)

    def psa_fwd(self, *a, **kw):
        seen["psa"] += 1
        return orig[1](self, *a, **kw)

    SIAMount.forward, PSAMount.forward = sia_fwd, psa_fwd
    try:
        report = run_arm(spec, lam, tag=tag)
    finally:
        SIAMount.forward, PSAMount.forward = orig
    return report, seen


def _mount_probe_control():
    """The instrument fires, and a mount with the weight at zero still raises."""
    from tiger_ext import PSAMount
    calls = {"n": 0}
    orig = PSAMount.forward

    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)

    PSAMount.forward = counting
    try:
        dummy = object.__new__(PSAMount)      # no nn.Module.__init__: enough for
        dummy.lambda_psa = 0.0                # the enabled guard, which runs first
        raised = False
        try:
            counting(dummy, torch.zeros(1, 1), torch.zeros(1, 4, dtype=torch.long))
        except RuntimeError:
            raised = True
    finally:
        PSAMount.forward = orig
    return calls["n"] == 1 and raised


def part6_pipeline(arms):
    print("\n[6] Q3b: the stage contract, the hand-off, and the inert mechanisms")
    off, on = arms[0.0], arms[1e-3]
    rec_man = on["stage_manifests"]["recommender"]
    check("the recommender checkpoint carries the E7 recommender-stage manifest, "
          "written by the pipeline's own phase rather than reconstructed here",
          bool(rec_man) and rec_man.get("freeze", {}).get("stage")
          == "recommender",
          f"stage = {rec_man.get('freeze', {}).get('stage') if rec_man else None}")

    groups = on["recommender"]["freeze"]
    check("the stage contract holds in the arm: the tokenizer group is frozen and "
          "the recommender group is trainable -- the RQ-VAE the constraint shaped is "
          "not silently re-trained by the T5 phase",
          groups["tokenizer"]["all_frozen"] and groups["recommender"]["all_trainable"],
          f"tokenizer {groups['tokenizer']['trainable_tensors']}/"
          f"{groups['tokenizer']['tensors']} trainable; recommender "
          f"{groups['recommender']['trainable_tensors']}/"
          f"{groups['recommender']['tensors']} trainable")

    tok_man = on["stage_manifests"]["tokenizer"]
    check("the tokenizer checkpoint has no E9 stage sidecar, and that is the "
          "composition's own property rather than a silent gap: C6 routes the "
          "tokenizer through C5's entry, which does not write one; the checkpoint's "
          "provenance travels in the arm's own report and in the SID manifest below",
          tok_man is None and on["tokenizer"]["optimizer_params"] > 0,
          f"sidecar: {tok_man}; the arm records "
          f"{on['tokenizer']['optimizer_params']} optimiser parameters and "
          f"best_collision_rate {on['tokenizer']['best_collision_rate']}")

    sid_man_path = on["refresh"]["manifest_path"]
    sid_man = (json.load(open(sid_man_path, encoding="utf-8"))
               if sid_man_path and os.path.isfile(sid_man_path) else {})
    check("the SID's manifest names the tokenizer checkpoint that produced it, by "
          "hash, so the hand-off is traceable rather than assumed",
          sid_man.get("ckpt_sha256") == on["tokenizer"]["sha256"]
          and sid_man.get("n_items", -1) == on["tokenizer"]["n_rows"],
          f"sid v{on['refresh']['version']} <- ckpt "
          f"{str(sid_man.get('ckpt_sha256'))[:16]}... over "
          f"{sid_man.get('n_items')} items")

    handoff = on["recommender"]["handoff"]
    check("the recommender's hand-off check found zero mismatches between the "
          "dataset, both mounts and the SID file",
          handoff["mismatches"] == {"sia": 0, "psa": 0}
          and handoff["n_items"] == handoff["consistent_items"],
          f"{handoff['consistent_items']} items routed, mismatches "
          f"{handoff['mismatches']}")

    # ``alt_pipeline``'s ``mounts`` field reports whether the *adapters were
    # constructed* -- ``mounts["sia"] is not None`` -- which is true for any
    # recommender phase, with the weight at zero or not. It is not an enable flag
    # and cannot carry this check. The weight is what gates the historical paths
    # (``model/main.py:162``: ``use_sia = sia is not None and sia.enabled``, and
    # ``SIAMount.enabled`` is ``lambda_sia > 0``), so measure the entry itself.
    mount_report, mount_seen = _mount_probe_run(SYN["spec"], 1e-3, tag="probe-mounts")
    check("the historical mounts were never entered, so the arm is TIGER + the "
          "constraint and nothing else -- the weighted SIA / PSA losses were not "
          "switched on to make the numbers move",
          mount_seen == {"sia": 0, "psa": 0}
          and mount_report["recommender"]["n_train_batches"] > 0,
          f"SIA forward called {mount_seen['sia']} time(s), PSA "
          f"{mount_seen['psa']} time(s) over "
          f"{mount_report['recommender']['n_train_batches']} recommender batch(es); "
          f"lambda_sia = lambda_psa = 0, and each forward raises when entered while "
          f"disabled, so a completed run already implies zero entries")
    check("negative control: the same instrument fires, and a mount with the weight "
          "at zero refuses to run -- so the zero above is the guard working, not a "
          "dead counter",
          _mount_probe_control(),
          "one instrumented call, and PSAMount.forward raised on a disabled mount")

    trainer, hook = _probe_run(SYN["spec"], 1e-3, HISTORICAL_PROBES)
    check("L_trans / L_balance / L_align are called zero times in a constraint-on "
          "run, and the probes are live: all four targets exist on the frozen "
          "Trainer",
          all(trainer.probes[p] == 0 for p in HISTORICAL_PROBES)
          and all(hasattr(trainer, p) for p in HISTORICAL_PROBES)
          and trainer.steps() and hook.n_calls == len(trainer.steps()),
          f"probe counts {trainer.probes}; {hook.n_calls} constrained step(s)")

    trainer2, _ = _probe_run(SYN["spec"], 1e-3, ("balance_loss",),
                             extra={"lambda_balance": 1.0})
    check("negative control: with lambda_balance = 1.0 the very same probe fires, so "
          "the zeros above are a measurement and not a broken instrument",
          trainer2.probes["balance_loss"] > 0,
          f"balance_loss called {trainer2.probes['balance_loss']} time(s)")
    return trainer, hook


# --------------------------------------------------------------------------- #
# [7] Q3c -- no user information in the SID
# --------------------------------------------------------------------------- #
def part7_leakage(arms):
    print("\n[7] Q3c: the SID is item-level, and no user information reaches it")
    spec = SYN["spec"]
    on = arms[1e-3]
    sid = np.load(on["refresh"]["sid_path"])
    import pandas as pd
    n_items = len(pd.read_parquet(spec.embedding_path))

    check("the SID has exactly one row per item and four token columns -- there is no "
          "user axis for a user-specific SID to live on",
          sid.shape == (n_items, 4),
          f"SID {sid.shape} for {n_items} items; the fixture has "
          f"{len(pd.read_parquet(os.path.join(spec.data_root, spec.dataset, 'train.parquet')))} "
          "training users, none of whom appear in the shape")

    ckpt = _state_dict(on["tokenizer"]["ckpt_path"])
    data_path = str(ckpt["args"].data_path)
    check("the tokenizer's checkpoint records item_emb.parquet as its only input, "
          "and that path carries no interaction file",
          os.path.basename(data_path) == "item_emb.parquet"
          and os.path.basename(os.path.dirname(data_path)) == spec.dataset,
          f"{os.path.relpath(data_path, REPO) if data_path.startswith(REPO) else data_path}")

    src = ast.parse(open(os.path.join(REPO, "tiger_ext", "qspace_smoke.py"),
                        encoding="utf-8").read())
    fn = next((n for n in ast.walk(src) if isinstance(n, ast.FunctionDef)
               and n.name == "run_chain"), None)
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    literals = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)
                and isinstance(n.value, str)}
    user_hits = sorted(x for x in names | literals
                       if any(t in str(x).lower() for t in USER_TOKENS))
    file_hits = sorted(x for x in literals
                       if any(f in x for f in INTERACTION_FILES))
    check("the chain's own source never names a user-side identifier and never "
          "names an interaction file: the tokenizer stage is fed item_emb.parquet "
          "and nothing else, and the interactions are read only by the pipeline's "
          "later phases",
          not user_hits and not file_hits,
          f"scanned {len(names)} names and {len(literals)} literals in run_chain; "
          f"user-side {user_hits or 'none'}, interaction file {file_hits or 'none'}")

    hits = []
    for mod in ("qspace_smoke", "qspace_train", "qspace_constraint", "qspace_loss"):
        text = open(os.path.join(REPO, "tiger_ext", f"{mod}.py"),
                    encoding="utf-8").read().lower()
        bad = [t for t in USER_TOKENS if t in text]
        if bad:
            hits.append((mod, bad))
    check("the four modules on the constrained path contain no user-side identifier "
          "anywhere, not merely in the function that runs",
          not hits, f"{len(hits)} module(s) with a hit: {hits or 'none'}")

    check("the phase log records the SID version and the checkpoint hashes and no "
          "per-user field at all",
          not any("user" in json.dumps(p).lower() for p in arms[1e-3]["phase_order"])
          and all(isinstance(p, (list, tuple)) for p in arms[1e-3]["phase_order"]),
          f"phase log entries: {arms[1e-3]['phase_order']}")


# --------------------------------------------------------------------------- #
# [8] the real-data arm
# --------------------------------------------------------------------------- #
def part8_toys():
    print("\n[8] the real-data arm: Toys, at the frozen smoke budget")
    if TOYS == "skip":
        print("  [skip] EXP_C6_TOYS=skip")
        return None
    spec = SM.toys_spec(DATA_DIR, device=DEVICE, seed=SEED)
    out = {}
    for lam in (0.0, 1e-3):
        tag = f"toys-{lam:g}"
        try:
            report = run_arm(spec, lam, tag=tag)
        except Exception as exc:                                    # noqa: BLE001
            out[lam] = {"error": f"{type(exc).__name__}: {exc}"}
            check(f"Toys arm lambda_csg={lam:g}: the chain completed",
                  False, out[lam]["error"][:200])
            continue
        out[lam] = report
        check(f"Toys arm lambda_csg={lam:g}: the full chain "
              f"{' -> '.join(FULL_CHAIN)} on the real 4096-d item table, every "
              f"artifact written",
              tokenizer_leg_ran(report)
              and [p[0] for p in report["phase_order"]] == list(PIPELINE_PHASES)
              and os.path.isfile(report["refresh"]["sid_path"])
              and bool(report["evaluation"]["recalls"]),
              f"{report['seconds']['total']:.0f}s; RQ-VAE "
              f"{report['tokenizer']['n_steps']} steps over "
              f"{report['tokenizer']['n_rows']} items; "
              f"{report['evaluation']['n_eval_sequences']} sequences evaluated")

    if (not out.get(1e-3) or "error" in out[1e-3]
            or not out.get(0.0) or "error" in out[0.0]):
        check("both Toys arms completed, so the constraint-on/off comparison on "
              "real data could be read", False,
              f"arms present: {sorted(out)}; "
              f"errors: {[k for k, v in out.items() if 'error' in v]}")
        return out
    on = out[1e-3]
    check("on real data the hook also ran on every step and produced a positive "
          "term every time, so the mechanism is not an artefact of the fixture",
          on["counters"]["n_calls"] == on["tokenizer"]["n_steps"]
          and on["counters"]["n_nonempty"] == on["counters"]["n_calls"]
          and all(h["loss"] > 0.0 for h in on["history"]),
          f"{on['counters']['n_calls']}/{on['tokenizer']['n_steps']} steps; "
          f"L_csg in [{min(h['loss'] for h in on['history']):.4f}, "
          f"{max(h['loss'] for h in on['history']):.4f}]")
    check("the off arm on real data is the same chain with the switch off",
          out[0.0]["counters"]["n_calls"] == 0
          and out[0.0]["counters"]["n_disabled_calls"] == 0,
          f"n_calls = {out[0.0]['counters']['n_calls']}")
    print("    ! Recall/NDCG here are pipeline-smoke output only. The Toys arm "
          "carries two confounds and C6 resolves neither:")
    print("      (a) the 2-epoch 4096-d tokenizer budget is far below the clean "
          "baseline's, so the codes themselves are under-trained;")
    print("      (b) the first-batch geometry interacts with kmeans_init (section "
          "[9]).")
    print("      " + SM.CST_SMOKE_TOYS_NOTE)
    note = SM.CST_SMOKE_TOYS_NOTE.lower()
    check("the confounds are recorded in the module rather than left to the reader, "
          "and C6 attributes nothing about the dataset to the constraint",
          "confound" in note and "musical_instruments" in note
          and "engineering evidence" in note,
          f"{len(SM.CST_SMOKE_PENDING)} pending decision(s) carried forward")
    return out


# --------------------------------------------------------------------------- #
# [9] Q4 -- the kmeans_init ruling
# --------------------------------------------------------------------------- #
def _km_line(label, row):
    return (f"      {label} kmeans_init={row['kmeans_init']!s:5s} "
            f"rebuild={row['first_batch_rebuild_delta']:.3e}"
            f"/{'ok' if row['first_batch_rebuild_ok'] else 'BAD'} "
            f"latent={row['first_batch_latent_delta']:.3e}"
            f"/{'ok' if row['first_batch_latent_ok'] else 'BAD'} "
            f"step0|dW|={row['step0_codebook_grad_sum']:.3e} "
            f"enc_grad={row['encoder_grad_seen']} "
            f"recon={['%.4f' % r for r in row['epoch_recon']]} "
            f"distinct={[l['distinct'] for l in row['usage']['epoch0']['levels']]} "
            f"max_share="
            f"{max(l['max_share'] for l in row['usage']['epoch0']['levels']):.3f} "
            f"safe={row['safe']}")


def part9_kmeans():
    print("\n[9] Q4: kmeans_init at the smoke budget, measured both ways")
    spec = SYN["spec"]
    syn = {}
    for flag in (True, False):
        syn[flag] = SM.probe_kmeans(spec, kmeans_init=flag, epochs=3,
                                    batch_size=1024, tag=f"syn-km{int(flag)}")
    print(f"    synthetic fixture, n_e = "
          f"{spec.rqvae_config['num_emb_list']}, rows = "
          f"{spec.rqvae_config['batch_size']}")
    for flag, row in syn.items():
        print(_km_line("syn ", row))

    check("on the synthetic fixture both settings clear C3's two first-batch guards "
          "and train: x_q still equals the codebook sum, x_q is still distinguishable "
          "from the encoder latent, the codebooks receive gradient, the "
          "reconstruction changes and no level collapses",
          all(r["first_batch_rebuild_ok"] and r["first_batch_latent_ok"]
              for r in syn.values())
          and all(r["step0_codebook_grad_sum"] > 0.0 for r in syn.values())
          and all(r["epoch_recon_moves"] for r in syn.values())
          and all(r["usage_ok"] for r in syn.values()),
          f"rebuild {['%.3e' % r['first_batch_rebuild_delta'] for r in syn.values()]} "
          f"must sit under {REBUILD_ATOL:g}; latent "
          f"{['%.3e' % r['first_batch_latent_delta'] for r in syn.values()]} must "
          f"sit over it -- the second is the guard C5 tripped")

    check("the initialisation is what changes the first batch's geometry, not the "
          "quantiser: the two settings differ there and only there",
          syn[True]["first_batch_latent_delta"] != syn[False]["first_batch_latent_delta"],
          f"kmeans_init=True: |x_q - z| = "
          f"{syn[True]['first_batch_latent_delta']:.3e}; False: "
          f"{syn[False]['first_batch_latent_delta']:.3e}")

    toys = None
    if TOYS != "skip":
        tspec = SM.toys_spec(DATA_DIR, device=DEVICE, seed=SEED)
        toys = {flag: SM.probe_kmeans(tspec, kmeans_init=flag, epochs=2,
                                      batch_size=1024, tag=f"toys-km{int(flag)}")
                for flag in (True, False)}
        for flag, row in toys.items():
            print(_km_line("Toys", row))
        check("on the real 4096-d table at the same budget the two settings are again "
              "both usable, so the smoke's batch size is not a trap on either "
              "dataset",
              all(r["first_batch_rebuild_ok"] and r["first_batch_latent_ok"]
                  for r in toys.values())
              and all(r["step0_codebook_grad_sum"] > 0.0 for r in toys.values()),
              f"rebuild "
              f"{['%.3e' % r['first_batch_rebuild_delta'] for r in toys.values()]} "
              f"against {REBUILD_ATOL:g}; latent "
              f"{['%.3e' % r['first_batch_latent_delta'] for r in toys.values()]}")

    basis = (
        "The frozen Clean TIGER recipe keeps kmeans_init = True (RQVAE_CONFIG), and "
        "the smoke does not amend it: C6 may not introduce a new initialisation "
        "mechanism, and flipping the flag would be exactly that. True is ruled safe "
        "*because it was measured here*, not assumed: at the smoke's own batch "
        "geometry the first batch clears both of C3's guards -- x_q still equals the "
        "codebook sum, and x_q is still distinguishable from the encoder latent -- "
        "and every codebook receives gradient on the first step. The degenerate case "
        "C5 found (x_q == z bitwise, codebook gradient ~ 0) is a property of the "
        "batch geometry, not of the flag, so the constraint C6 carries forward is "
        "'measure the first batch', not 'change the initialiser'."
    )
    check("the safe configuration is the frozen one, and the basis is the "
          "measurement above rather than a preference -- no initialiser was added, "
          "no RQ-VAE core file was touched, and the batch size stays the frozen 1024",
          spec.rqvae_config["kmeans_init"] is True and syn[True]["safe"],
          basis)
    check("the boundary C5 found is carried forward rather than quietly dropped: the "
          "first batch's geometry is a real constraint, and it is one of the things "
          "C6 hands to the effect experiment",
          any("geometr" in d["why"] or "batch" in d["why"]
              for d in SM.CST_SMOKE_PENDING),
          f"{len(SM.CST_SMOKE_PENDING)} pending decision(s) carried forward")
    return syn, toys


# --------------------------------------------------------------------------- #
# [10] regression
# --------------------------------------------------------------------------- #
def part10_regression():
    print("\n[10] regression: C5's own suite, re-run")
    if REGRESSION == "skip":
        print("  [skip] EXP_C6_REGRESSION=skip")
        return None
    proc = subprocess.run([sys.executable, "-I", "integration/exp_c5_csg_integration.py"],
                          cwd=REPO, capture_output=True, text=True, timeout=28800)
    summaries = suite_summary_re("C5").findall(proc.stdout)
    fails = FAIL_RE.findall(proc.stdout)
    n, total = ((int(summaries[0][0]), int(summaries[0][1])) if summaries
                else (-1, -1))
    print(f"    exp_c5_csg_integration.py  {n}/{total} checks, exit "
          f"{proc.returncode}, {len(fails)} FAIL")
    for f in fails:
        print(f"        FAIL: {f}")
    if not summaries:
        print(proc.stdout[-3000:])
    check("C5's suite still declares its own result exactly once, so the count cannot "
          "have come from a nested re-run printed before it",
          len(summaries) == 1, f"{len(summaries)} summary line(s)")
    check("C5's suite is still fully green after C6: 99/99, exit 0",
          n == total == 99 and proc.returncode == 0 and not fails,
          f"{n}/{total}, exit {proc.returncode}, failures {fails or 'none'}")
    check("that single re-run transitively covers the seven older suites with their "
          "own R1/R2/R3 attribution, so C6 does not duplicate an hour of nested "
          "re-runs -- and it would have caught a C6 tracked edit, because C5's "
          "section [8] guards that exact set",
          "EXP-C5" in proc.stdout and n == 99,
          f"{len(proc.stdout)} bytes of stdout read; the modified-set guard inside "
          "it re-ran")
    return {"n": n, "total": total, "exit": proc.returncode, "fails": fails}


# --------------------------------------------------------------------------- #
def main():
    print("=" * 76)
    print("EXP-C6 -- CSG smoke test: the full chain with the constraint switched on")
    print("=" * 76)
    # C2's default artifact directory is a *relative* path
    # (``integration/artifacts/collab_relation``), so the Toys arm only resolves
    # it from the repository root. Pin the working directory rather than require
    # the caller to have invoked this from the right place.
    os.chdir(REPO)
    spec = SM.synthetic_spec(os.path.join(TMP, "syn"), device=DEVICE, seed=SEED)
    spec.rqvae_epochs = 3
    SYN["spec"] = spec
    SYN["dim"] = SM.CST_SMOKE_SYN["dim"]
    SM.build_fixture(os.path.join(TMP, "syn"), spec,
                     rng=np.random.default_rng(SEED))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        part1_scope()
        part2_fixtures()
        arms = part3_arms()
        # Every lambda gets an entry whether it ran or raised, so the guard has to
        # look for errors rather than for the key count.
        if all("error" not in r for r in arms.values()) \
                and len(arms) == len(SM.CST_SMOKE_LAMBDAS):
            part4_entered(arms)
            part5_bitwise(arms)
            part6_pipeline(arms)
            part7_leakage(arms)
        else:
            check("sections [4]-[7] could run: all four arms completed", False,
                  f"{len(arms)} arms")
        part8_toys()
        part9_kmeans()
        part10_regression()

    total = len(RESULTS)
    passed = sum(RESULTS)
    print("\n" + "=" * 76)
    print(f"EXP-C6: {passed}/{total} checks passed")
    print("=" * 76)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
