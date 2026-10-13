# Clean TIGER baseline

A frozen, isolated copy of the **original TIGER pipeline**, with every historical
experiment mechanism switched off. It exists so that later experiments have a
stable reference point to compare against.

```
item content representation  →  RQ-VAE  →  Semantic ID  →  T5  →  Recommendation
```

No code was deleted and no file was moved: this is a *configuration and entry
point*, not a fork. Every historical experiment still runs exactly as before.

---

## Quick start

```bash
bash clean_tiger/run_clean_tiger.sh all          # full pipeline, DATASET=Toys
DATASET=Beauty bash clean_tiger/run_clean_tiger.sh all
bash clean_tiger/smoke_test.sh                   # ~minutes, tiny budgets
```

Individual stages:

```bash
bash clean_tiger/run_clean_tiger.sh rqvae    # 1. RQ-VAE
bash clean_tiger/run_clean_tiger.sh sid      # 2. Semantic IDs
bash clean_tiger/run_clean_tiger.sh t5       # 3. T5 recommender
bash clean_tiger/run_clean_tiger.sh eval     # 4. evaluation
bash clean_tiger/run_clean_tiger.sh verify   # prove nothing historical loads
```

All hyper-parameters live in [`config.sh`](config.sh); override any of them from
the environment.

---

## What "Clean" means here

### Included (original TIGER)

| Stage | Code | Input |
|---|---|---|
| Item content representation | `data/<D>/item_emb.parquet` (Sentence-T5) | item metadata |
| RQ-VAE | `rqvae/main.py`, `rqvae/models/*`, `rqvae/trainer.py` | `item_emb.parquet` |
| Semantic ID | `rqvae/generate_code.py` | RQ-VAE checkpoint |
| T5 recommender | `model/main.py`, `model/dataset.py`, `model/dataloader.py` | `<D>_clean_tiger_sid.npy` + `train/valid/test.parquet` |

`model/` (the whole T5 side) is byte-for-byte the state of the initial import —
it has never been touched by any experiment.

### Excluded (historical experiments), and how they are kept out

| Mechanism | Where it lives | How it is kept out of Clean TIGER |
|---|---|---|
| behavior embedding (PPMI / SVD, 128-d) | `rqvae/behavior_variants.py`, `data/<D>/behavior_emb*.parquet` | input file is `item_emb.parquet`, never `fused_item_emb*` |
| content + behavior fusion | `rqvae/fuse_embeddings.py` | same — `assert_clean_emb()` refuses `fused_item_emb*` |
| transition / co-occurrence graph | `rqvae/build_transition_graph.py` | `--lambda_trans 0.0`, `--trans_graph_path` unset |
| `L_trans` (Track B contrastive) | `rqvae/trans_loss.py`, `trainer.py:_get_transition_loss` | `--lambda_trans 0.0` |
| `L_balance` (Track B codebook balance) | `rqvae/trans_loss.py`, `trainer.py` | `--lambda_balance 0.0` |
| `L_align` (E50 b2 InfoNCE) | `trainer.py:_get_align_loss`, `--align_weight` | `--align_weight 0.0`, `--align_data_path` unset |
| TA-SID / Track B / E50 drivers | `rqvae/run_ta_sid_*.py`, `run_b2_*.sh`, `tools/*` | not invoked by this entry point |

The three zeros are passed **explicitly** on the command line rather than relied
on as defaults, so a future edit to a default cannot silently re-enable a
mechanism. See `rqvae_clean_args()` in `config.sh`.

`clean_tiger/verify_clean.py` proves this at runtime rather than by assertion:
it instruments every `read_parquet`/`np.load`, trips wires on the three loss
terms, runs real training steps, and checks that none of them fired and that no
behavior/fused/transition file was read.

---

## Isolation of outputs

Everything this pipeline writes goes under `clean_tiger/`:

```
clean_tiger/ckpt/<DATASET>/rqvae/<timestamp>/best_collision_model.pth
clean_tiger/ckpt/<DATASET>/tiger_clean_<DATASET>_seed<S>.pth
clean_tiger/logs/
clean_tiger/_smoke/            # smoke-test artefacts, safe to delete
```

No historical checkpoint, log or Semantic ID file is overwritten.

---

## Calling the scripts directly (cwd matters)

`rqvae/main.py` and `rqvae/generate_code.py` do `from datasets import EmbDataset`,
which resolves to the local `rqvae/datasets.py` **only when `rqvae/` is on
`sys.path`**. Always run them from inside `rqvae/`:

```bash
cd rqvae && python main.py --data_path ../data/Toys/item_emb.parquet ...
```

Running them from the repo root picks up the HuggingFace `datasets` package
instead and fails with `ImportError: cannot import name 'EmbDataset'`. The
scripts here already `cd` correctly.

---

## Known divergences from upstream TIGER (recorded, inert)

Two changes were made to `rqvae/` by later experiments. Neither affects Clean
TIGER results at the configuration above; both are recorded here so the frozen
baseline is honest about them.

1. **`RQVAE.forward` default is now `return_z=True`** (returns a 4-tuple) where
   upstream returned 3. **Inert**: every caller passes `return_z` explicitly or
   uses `get_indices()`; no Clean TIGER path uses the default. Reported as
   `[INFO]` by `verify_clean.py`.

2. **`models/layers.py: kmeans()` has a `num_clusters > B` guard** (added for the
   K0=1536/2048 sweeps). **Inert**: with the Clean TIGER default
   `num_emb_list=[256,256,256]` and `batch_size=1024`, `256 > 1024` is false, so
   the branch is never taken.

`rqvae/datasets.py` gained an optional `return_index` flag (default `False`),
which is likewise inactive here — it only exists for row-index lookup used by
`L_trans` / `L_align`.

---

## Smoke test coverage

`smoke_test.sh` runs the **real** stages — not mocks:

1. **RQ-VAE** — 2 epochs on the full content-only embedding (real `item_emb.parquet`).
2. **Semantic ID** — full pass over all items, producing a real `<D>_clean_tiger_sid.npy`.
3. **T5** — 1 epoch on a 200-user truncated interaction set.
4. **Evaluation** — beam search + Recall/NDCG on the truncated test split.
5. **Verification** — the contamination check above.

Only the interaction parquets are truncated; the item embedding, the RQ-VAE and
the Semantic ID stage are genuinely exercised end to end.
