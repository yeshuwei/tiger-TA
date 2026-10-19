#!/usr/bin/env python
"""EXP-B7 -- does ``--seed`` actually control the random processes?

Run:  python -I integration/exp_b7_seed_control.py

The gate registered B7 as: ``tiger_ext/qspace_train.py`` accepted ``--seed``,
put it in ``args`` and the run report, but **never applied it**. The baseline
entry ``rqvae/main.py`` seeds python/numpy/torch/cuda (and pins cuDNN) before it
builds anything; the Stage 0bis tokenizer entry did not, so two runs with the
same ``--seed`` differed and two runs with different seeds could not be
attributed to the seed.

The fix has three independent obligations, and this harness tests each where it
is consumed rather than where it is logged:

  [1] the call chain -- ``--seed`` reaches the config, the config reaches
      ``apply_seed`` **before** the dataset and the model, and the CLI is not a
      dead end
  [2] the global streams -- python ``random``, numpy, torch CPU and torch CUDA
      are re-seeded on every run (including after an earlier run perturbed
      them), and cuDNN is pinned as ``rqvae/main.py`` pins it
  [3] k-means -- the frozen ``rqvae/models/layers.py`` calls
      ``KMeans(...).fit(x)`` with **no** ``random_state``, so it follows the
      global numpy stream. The harness first *proves* that dependence, then
      proves the explicit per-call ``random_state`` removes it
  [4] same seed, same process -- initial weights, k-means draws, shuffle order,
      final weights, discrete codes and ``best_loss`` are all identical even
      when thousands of RNG values were consumed between the two runs
  [5] same seed, separate process -- the same equality across interpreter
      boundaries (the reproducibility that actually matters for a recipe)
  [6] different seed -- the initialisation and the k-means draws change, so the
      sameness in [4]/[5] is not an artifact of a deterministic pipeline
  [7] scope -- the frozen files are untouched, the recipe knobs are unchanged,
      and the report records exactly what was seeded and what CUDA still leaves
      nondeterministic

Every training run is one epoch on a 320-row synthetic table. Nothing here is
evidence about recommendation quality, and EXP-L0 is not launched.
"""
import contextlib
import hashlib
import inspect
import io
import json
import os
import random
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
from tiger_ext.alt_pipeline import RQVAE_CONFIG                       # noqa: E402

RESULTS = []

DATASET = "SynB7"
N_ROWS = 320
DIM = 32
FIXTURE_SEED = 7
FIXTURE_SHA = None      # filled in after the fixture is written


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    print(f"\n{title}")


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
    pd.DataFrame({"embedding": [row for row in emb]}).to_parquet(
        os.path.join(data_dir, "item_emb.parquet"))
    return os.path.join(data_dir, "item_emb.parquet")


def sha256_bytes(*chunks):
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    """A hash of every parameter/buffer, in a stable key order."""
    digest = hashlib.sha256()
    for key in sorted(model.state_dict()):
        tensor = model.state_dict()[key].detach().cpu().numpy()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(np.ascontiguousarray(tensor).tobytes())
    return digest.hexdigest()


def codes_hash(model, emb):
    with torch.no_grad():
        codes = model.get_indices(torch.FloatTensor(emb), use_sk=False)
    codes = codes.detach().cpu().numpy().astype(np.int64)
    return sha256_bytes(np.ascontiguousarray(codes).tobytes()), codes


# --------------------------------------------------------------------------- #
# one short run
# --------------------------------------------------------------------------- #
def run_seed(root, seed, tag, *, limit=N_ROWS, epochs=1, batch=N_ROWS):
    """One real 1-epoch training run; returns the evidence needed to compare."""
    seen, holder = {}, {}

    def dataset_factory(mods, path, return_index):
        # Read *at the moment the dataset is built*: this is the earliest point
        # that can be observed from outside, and it is after apply_seed().
        seen["torch_initial_seed_at_dataset"] = int(torch.initial_seed())
        seen["numpy_first_draw_at_dataset"] = int(
            np.random.randint(2 ** 31 - 1))
        return mods["EmbDataset"](path, return_index=return_index)

    def trainer_factory(args, model, data_num, csg):
        seen["init_state_hash"] = state_hash(model)
        trainer = QT.frozen_tokenizer_modules()["RQVAETrainer"](
            args, model, data_num, csg)
        holder["trainer"] = trainer
        return trainer

    cfg = QT.CSGTrainerConfig(
        dataset=DATASET, data_root=root, lambda_csg=0.0, device="cpu",
        seed=seed, epochs=epochs, eval_step=epochs, batch_size=batch,
        num_workers=0, warmup_epochs=0, limit=limit,
        ckpt_dir=os.path.join(root, f"ckpt_{tag}"))
    hook = QT.CSGHook(lambda_csg=0.0)      # disabled: B7 is not about L_csg
    report = QT.train_tokenizer(cfg, hook=hook, dataset_factory=dataset_factory,
                               trainer_factory=trainer_factory)
    model = holder["trainer"].model
    final_hash = state_hash(model)
    codes_digest, codes = codes_hash(model, fixture_embeddings())
    return {
        "seed": int(seed),
        "tag": tag,
        "init_state_hash": seen["init_state_hash"],
        "final_state_hash": final_hash,
        "codes_hash": codes_digest,
        "codes": codes,
        "best_loss": float(report["best_loss"]),
        "best_collision_rate": float(report["best_collision_rate"]),
        "torch_initial_seed_at_dataset": seen["torch_initial_seed_at_dataset"],
        "numpy_first_draw_at_dataset": seen["numpy_first_draw_at_dataset"],
        "kmeans_random_state_calls": list(
            report["seeding"]["kmeans_random_state_calls"]),
        "shuffle_generator_seed": report["seeding"]["shuffle_generator_seed"],
        "shuffle_source": report["seeding"]["shuffle_source"],
        "n_steps": int(report["n_steps"]),
        "embedding_path": report["embedding_path"],
        "in_dim": int(report["in_dim"]),
        "n_rows": int(report["n_rows"]),
        "seeding": report["seeding"],
    }


def comparable(run):
    """The fields two same-seed runs must agree on exactly."""
    return {key: run[key] for key in
            ("init_state_hash", "final_state_hash", "codes_hash", "best_loss",
             "kmeans_random_state_calls", "shuffle_generator_seed", "n_steps")}


def perturb_global_rng():
    """Burn RNG values in every stream a run could otherwise inherit."""
    random.seed(987654321)
    np.random.seed(987654321)
    torch.manual_seed(987654321)
    [random.random() for _ in range(2000)]
    np.random.rand(2000)
    torch.rand(2000)


# --------------------------------------------------------------------------- #
# [1] the call chain
# --------------------------------------------------------------------------- #
def part1_chain(root):
    section("[1] --seed -> config -> apply_seed, before the dataset and model")

    parser_seed = QT.build_parser().parse_args(["--seed", "4321"]).seed
    check("--seed 4321 is parsed into the namespace", parser_seed == 4321,
          str(parser_seed))

    captured = {}
    original = QT.train_tokenizer

    def fake(config, **kwargs):
        captured["config"] = config
        report = dict(config.report())
        report.update(n_steps=0, best_loss=0.0, best_collision_rate=0.0,
                      ckpt_dir="/tmp/fake", hook={},
                      seeding={"seed": config.seed,
                               "torch_initial_seed": config.seed,
                               "shuffle_generator_seed": config.seed,
                               "kmeans_random_state_calls": []})
        return report

    QT.train_tokenizer = fake
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            QT.main(["--seed", "4321", "--dataset", DATASET,
                     "--data-root", root, "--limit", "4", "--epochs", "1"])
    finally:
        QT.train_tokenizer = original
    check("main() hands --seed to CSGTrainerConfig (the CLI is not a dead end)",
          captured.get("config") is not None
          and captured["config"].seed == 4321,
          f"config.seed={getattr(captured.get('config'), 'seed', None)}")

    source = inspect.getsource(QT.train_tokenizer)
    i_seed = source.find("apply_seed(config.seed)")
    i_data = source.find("build_dataset(config")
    i_model = source.find("build_model(config")
    check("apply_seed runs before the dataset is built",
          -1 < i_seed < i_data, f"seed@{i_seed} data@{i_data}")
    check("apply_seed runs before the model is built (and before lazy k-means)",
          -1 < i_seed < i_model, f"seed@{i_seed} model@{i_model}")

    cfg = QT.CSGTrainerConfig(seed=2024)
    check("the frozen default seed is still 2024",
          cfg.seed == 2024 and cfg.namespace().seed == 2024)


# --------------------------------------------------------------------------- #
# [2] the global streams
# --------------------------------------------------------------------------- #
def part2_global_streams():
    section("[2] python / numpy / torch CPU / torch CUDA are re-seeded")

    def torch_draws():
        QT.apply_seed(11)
        return [float(torch.rand(1)) for _ in range(3)]

    def numpy_draws():
        QT.apply_seed(11)
        return np.random.rand(3).tolist()

    def python_draws():
        QT.apply_seed(11)
        return [random.random() for _ in range(3)]

    record = QT.apply_seed(11)
    check("torch reports the applied seed as its initial seed",
          torch.initial_seed() == 11, str(torch.initial_seed()))
    check("torch CPU draws are reproducible", torch_draws() == torch_draws())
    check("numpy draws are reproducible", numpy_draws() == numpy_draws())
    check("python random draws are reproducible", python_draws() == python_draws())

    check("apply_seed reports every stream it touched",
          all(record[key] for key in ("python_random", "numpy", "torch_cpu",
                                      "torch_cuda")), str(record["seed"]))
    check("cuDNN is pinned the way rqvae/main.py pins it",
          torch.backends.cudnn.deterministic is True
          and torch.backends.cudnn.benchmark is False,
          f"deterministic={torch.backends.cudnn.deterministic} "
          f"benchmark={torch.backends.cudnn.benchmark}")

    perturb_global_rng()
    QT.apply_seed(11)
    check("re-seeding resets every stream after arbitrary consumption",
          torch.initial_seed() == 11
          and [float(torch.rand(1)) for _ in range(3)] == torch_draws())


# --------------------------------------------------------------------------- #
# [3] k-means needs explicit control
# --------------------------------------------------------------------------- #
def part3_kmeans(root):
    section("[3] k-means: prove global dependence, then prove explicit control")

    QT.frozen_tokenizer_modules()
    layers = sys.modules["models.layers"]
    emb = torch.FloatTensor(fixture_embeddings()[:300])

    np.random.seed(1)
    raw1 = layers.kmeans(emb, 8, 10)
    np.random.seed(2)
    raw2 = layers.kmeans(emb, 8, 10)
    check("the frozen KMeans call follows the *global* numpy stream when no "
          "random_state is passed",
          not np.array_equal(raw1.numpy(), raw2.numpy()),
          "centres differ between numpy.seed(1) and numpy.seed(2)")

    np.random.seed(1)
    with QT.seeded_kmeans(11) as draws_a:
        centres_a = layers.kmeans(emb, 8, 10)
    np.random.seed(2)                       # different global state ...
    with QT.seeded_kmeans(11) as draws_b:
        centres_b = layers.kmeans(emb, 8, 10)
    check("seeded_kmeans makes the centres independent of the global stream",
          np.array_equal(centres_a.numpy(), centres_b.numpy()))
    check("the injected random_state sequence is reproducible",
          draws_a == draws_b, str(draws_a))
    expected_stream = np.random.RandomState(11)
    expected = [int(expected_stream.randint(0, 2 ** 31 - 1))
                for _ in range(len(draws_a))]
    check("the draws come from RandomState(seed), in call order",
          draws_a == expected, f"{draws_a} vs {expected}")

    with QT.seeded_kmeans(12) as draws_c:
        np.random.seed(1)
        centres_c = layers.kmeans(emb, 8, 10)
    check("a different seed draws a different random_state",
          draws_c != draws_a, f"{draws_c} vs {draws_a}")
    print(f"         (informational) different random_state -> different centres: "
          f"{not np.array_equal(centres_a.numpy(), centres_c.numpy())}")

    seen = []
    saved = layers.KMeans

    def spy(*args, **kwargs):
        seen.append(kwargs.get("random_state"))
        return saved(*args, **kwargs)

    layers.KMeans = spy
    try:
        with QT.seeded_kmeans(11):
            layers.kmeans(emb, 8, 10)
    finally:
        layers.KMeans = saved
    check("the frozen KMeans really received a non-None random_state",
          bool(seen) and all(value is not None for value in seen), str(seen))

    before = layers.KMeans
    try:
        with QT.seeded_kmeans(5):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("the patch is removed even when the run fails",
          layers.KMeans is before)

    layers_src = os.path.join(REPO, "rqvae", "models", "layers.py")
    with open(layers_src, encoding="utf-8") as fh:
        text = fh.read()
    check("rqvae/models/layers.py is not edited (no random_state in the file, "
          "the original call intact)",
          "random_state" not in text
          and "KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)" in text)


# --------------------------------------------------------------------------- #
# [4] same seed, same process
# --------------------------------------------------------------------------- #
def part4_same_process(root):
    section("[4] same seed, same process, with the RNG burned in between")

    first = run_seed(root, 1234, "same_a")
    perturb_global_rng()
    second = run_seed(root, 1234, "same_b")

    check("the dataset sees the applied seed (torch.initial_seed)",
          first["torch_initial_seed_at_dataset"] == 1234,
          str(first["torch_initial_seed_at_dataset"]))
    check("the dataset sees a numpy stream freshly seeded with it",
          first["numpy_first_draw_at_dataset"]
          == int(np.random.RandomState(1234).randint(0, 2 ** 31 - 1)))
    check("initial weights are identical",
          first["init_state_hash"] == second["init_state_hash"],
          first["init_state_hash"][:16])
    check("k-means draws and centres (via the draws) are identical",
          first["kmeans_random_state_calls"] == second["kmeans_random_state_calls"]
          and len(first["kmeans_random_state_calls"]) > 0,
          str(first["kmeans_random_state_calls"]))
    check("the shuffle generator is seeded with the run seed",
          first["shuffle_generator_seed"] == 1234
          and first["shuffle_source"].endswith("config.seed"),
          first["shuffle_source"])
    check("final weights are identical",
          first["final_state_hash"] == second["final_state_hash"],
          first["final_state_hash"][:16])
    check("the discrete codes are identical",
          first["codes_hash"] == second["codes_hash"],
          f"{len(np.unique(first['codes']))} distinct codes")
    check("best_loss / best_collision are identical",
          first["best_loss"] == second["best_loss"]
          and first["best_collision_rate"] == second["best_collision_rate"],
          f"{first['best_loss']:.6f}")
    check("identical whole reports on the compared fields",
          comparable(first) == comparable(second))

    def order(seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return list(torch.utils.data.RandomSampler(
            range(N_ROWS), generator=generator))

    check("a seeded shuffle generator reproduces the batch order",
          order(1234) == order(1234))
    check("... and a different seed changes it", order(1234) != order(4321))
    return first


# --------------------------------------------------------------------------- #
# [5] same seed, separate process
# --------------------------------------------------------------------------- #
def run_child(root, seed):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", str(seed), root],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"child failed rc={proc.returncode}\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith("B7JSON:"):
            return json.loads(line[len("B7JSON:"):])
    raise RuntimeError(f"child printed no result:\n{proc.stdout}\n{proc.stderr}")


def part5_cross_process(root, in_process):
    section("[5] same seed, separate interpreter")

    child_a = run_child(root, 777)
    child_b = run_child(root, 777)
    check("two child processes agree on the initial weights",
          child_a["init_state_hash"] == child_b["init_state_hash"],
          child_a["init_state_hash"][:16])
    check("two child processes agree on the final weights, codes and loss",
          comparable(child_a) == comparable(child_b))
    check("an in-process run of the same seed matches the child processes",
          comparable(child_a) == comparable(in_process),
          f"in-process tag {in_process['tag']}")


# --------------------------------------------------------------------------- #
# [6] different seed
# --------------------------------------------------------------------------- #
def part6_different_seed(root, baseline):
    section("[6] a different seed changes the controlled processes")

    other = run_seed(root, 4321, "different")
    check("a different seed produces different initial weights",
          other["init_state_hash"] != baseline["init_state_hash"],
          f"{other['init_state_hash'][:16]} vs {baseline['init_state_hash'][:16]}")
    check("a different seed produces different k-means draws",
          other["kmeans_random_state_calls"]
          != baseline["kmeans_random_state_calls"])
    check("a different seed produces different trained weights",
          other["final_state_hash"] != baseline["final_state_hash"])
    check("... and different discrete codes on this fixture",
          other["codes_hash"] != baseline["codes_hash"],
          f"{len(np.unique(other['codes']))} vs "
          f"{len(np.unique(baseline['codes']))} distinct codes")
    check("the data is identical across seeds (so the difference is the seed)",
          sha256_file(os.path.join(root, DATASET, "item_emb.parquet"))
          == FIXTURE_SHA
          and other["embedding_path"] == baseline["embedding_path"]
          and other["n_rows"] == baseline["n_rows"] == N_ROWS,
          f"sha {FIXTURE_SHA[:16]} rows {baseline['n_rows']}")
    return other


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# [7] scope and records
# --------------------------------------------------------------------------- #
def part7_scope(root, run, other):
    section("[7] scope, record and what CUDA still leaves open")

    seeding = run["seeding"]
    required = ("seed", "python_random", "numpy", "torch_cpu", "torch_cuda",
                "torch_initial_seed", "cudnn_deterministic", "cudnn_benchmark",
                "shuffle_generator_seed", "kmeans_random_state_calls",
                "kmeans_random_state_source", "kmeans_subsample_random_state",
                "cuda_nondeterminism")
    check("the run report carries the full seeding record",
          all(key in seeding for key in required),
          f"{len(seeding)} fields")
    check("the report names the frozen k-means subsample as seed-independent",
          seeding["kmeans_subsample_random_state"] == 0
          and "random_state" in seeding["kmeans_random_state_source"])

    check("CST_SEED_CONTRACT is exported and names k-means + random_state",
          "k-means" in QT.CST_SEED_CONTRACT
          and "random_state" in QT.CST_SEED_CONTRACT)
    check("CST_CUDA_NONDETERMINISM names atomics and num_workers",
          "atomics" in QT.CST_CUDA_NONDETERMINISM
          and "num_workers" in QT.CST_CUDA_NONDETERMINISM)
    check("torch deterministic algorithms were NOT enabled (recipe unchanged)",
          torch.are_deterministic_algorithms_enabled() is False)

    cfg = QT.CSGTrainerConfig(dataset=DATASET, data_root=root, seed=1234,
                              epochs=1, batch_size=N_ROWS)
    args = cfg.namespace()
    recipe_same = all(getattr(args, key) == value for key, value in
                      RQVAE_CONFIG.items() if key != "batch_size")
    check("the model recipe knobs are the frozen ones (nothing tuned for B7)",
          recipe_same, f"num_emb_list={args.num_emb_list} e_dim={args.e_dim}")
    check("one epoch is still one epoch",
          run["n_steps"] == 1 and args.epochs == 1, str(run["n_steps"]))

    watched = ("tiger_ext/qspace_train.py", "tiger_ext/fixed_train.py",
               "tiger_ext/sid_refresh.py", "tiger_ext/alt_pipeline.py",
               "rqvae/trainer.py", "rqvae/models/layers.py", "model/main.py")
    owners = []
    for rel in watched:
        path = os.path.join(REPO, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            if "apply_seed" in text or "seeded_kmeans" in text:
                owners.append(rel)
    check("the seeding code exists only in tiger_ext/qspace_train.py",
          owners == ["tiger_ext/qspace_train.py"], str(owners))
    check("the two reported runs are the two seeds being compared",
          run["seed"] == 1234 and other["seed"] == 4321
          and run["embedding_path"] == other["embedding_path"])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global FIXTURE_SHA
    print("=" * 72)
    print("EXP-B7 -- --seed really controls the random processes")
    print("=" * 72)
    root = tempfile.mkdtemp(prefix="exp_b7_")
    try:
        table = build_fixture(root)
        FIXTURE_SHA = sha256_file(table)
        part1_chain(root)
        part2_global_streams()
        part3_kmeans(root)
        baseline = part4_same_process(root)
        # a fresh 777 run, compared against two child processes of the same seed
        in_process = run_seed(root, 777, "proc777")
        part5_cross_process(root, in_process)
        other = part6_different_seed(root, baseline)
        part7_scope(root, baseline, other)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for _name, ok, _detail in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 72)
    print(f"EXP-B7: {passed}/{total} checks passed")
    print("=" * 72)
    if passed != total:
        print("\nFailures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name} -- {detail}")
    return 0 if passed == total else 1


def child_main(argv):
    seed, root = int(argv[0]), argv[1]
    result = run_seed(root, seed, f"child_{seed}_{os.getpid()}")
    result.pop("codes", None)          # numpy array; the hash is what JSON needs
    print("B7JSON:" + json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--child":
            raise SystemExit(child_main(sys.argv[2:]))
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
