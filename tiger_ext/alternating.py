"""E9 -- the alternating Tokenizer / SID-refresh / Recommender scheduler.

Contract: ``integration/plan.md`` E9 table.

    | 需要新增什么 | 交替调度器 Tokenizer -> SID refresh -> Recommender -> ...; 每轮落盘 manifest |
    | 需要修改什么 | 无既有文件                                                          |
    | 输出         | 每轮的 tokenizer checkpoint + SID 版本 + recommender checkpoint      |
    | 验收标准     | (1) >=2 完整交替周期 (2) 每轮 ckpt/SID/指标链条可追溯
    |                (3) 中断可续跑     (4) 随机性有种子控制                          |

This module is the orchestration layer and **nothing else**. It owns:

  * the phase schedule        -- which epoch belongs to which stage
  * the cycle state machine   -- ``started -> tokenizer_done -> refresh_done ->
                                 recommender_done`` (+ an optional evaluation)
  * the resume decision       -- which step of which cycle to re-enter
  * the hand-off check        -- the SID the recommender actually consumed must
                                 be the SID this cycle's refresh produced

It does **not** own any mechanism. The four phase bodies are injected as
callables, so the scheduler can be exercised with fakes (branching, failure and
resume logic) entirely independently of training. :mod:`tiger_ext.alt_pipeline`
supplies the real ones for the TIGER port.

Where the rules come from
-------------------------
Every rule below is a transcription of the ETEGRec source, not an invention.
``reference_repos/ETEGRec_original/trainer.py``:

  * ``trainer.py:513`` ``for epoch_idx in range(self.epochs):`` -- the index is
    **0-based**, so epoch 0 is the first epoch.
  * ``trainer.py:515`` / ``trainer.py:542``
    ``if epoch_idx % self.cycle == 0:`` selects the tokenizer phase, ``else``
    the recommender phase. With ``config/scientific.yaml:28 cycle: 2`` the
    sequence is exactly ``T, R, T, R, ...`` -- one epoch per phase. ``cycle`` is
    a *modulo period*, not a block length: within every ``cycle`` epochs the
    first is the tokenizer and the remaining ``cycle - 1`` are recommender.
  * ``trainer.py:544`` ``all_item_code = self.get_code(...)`` runs immediately
    after every tokenizer epoch, and ``trainer.py:238-240`` /
    ``trainer.py:344-346`` feed ``self.all_item_code`` into the *next* epoch --
    so a new SID takes effect from the following recommender epoch, never
    mid-phase.
  * ``trainer.py:56`` ``assert self.cycle % self.eval_step == 0`` -- the
    evaluation cadence must divide the phase period. With
    ``cycle = eval_step = 2`` every evaluation lands at the end of a
    recommender epoch, i.e. **once per cycle, never after a tokenizer epoch**.

Deliberately not ported (already recorded as out of scope, ``plan.md`` §10 R4):
``warm_epoch = 10`` (a *loss-weight* warm-up on ``kl_loss`` / ``dec_cl_loss``),
the in-batch target de-duplication, and the official ``finetune()`` arm. This
module introduces no warm-up, no extra loss and no new parameter group.

The initial refresh
-------------------
The flow the plan and the Task spec fix is

    initial ckpt -> [initial SID refresh] -> (T -> refresh -> R) -> (T -> refresh -> R) -> ...

The initial refresh is **not** part of any cycle (the Task spec says so
explicitly) and is recorded separately in ``<run_dir>/initial_refresh.json``.
It mirrors ETEGRec's pre-loop ``get_code(epoch_idx=-1)`` (``trainer.py:510``),
which is likewise overwritten before the first recommender epoch: in the TIGER
port the tokenizer phase trains the RQ-VAE on the fixed content embedding and
never reads a SID at all, so the initial SID has no *training* consumer. Its
two jobs here are to root the version chain and to be the diff baseline of
cycle 0's refresh (E8 auto-discovers the newest registered version as
"previous", ``sid_refresh.py:415-418``).
"""
import datetime as _dt
import json
import os

from .sid_refresh import sha256_file
from .stage import STAGE_RECOMMENDER, STAGE_TOKENIZER, STAGES

# --------------------------------------------------------------------------- #
# The schedule
# --------------------------------------------------------------------------- #
#: One cycle = one tokenizer epoch + one recommender epoch (E9 Task spec §4).
CYCLE_LENGTH = 2

#: ETEGRec's ``config/scientific.yaml:35 eval_step: 2`` + ``trainer.py:56``.
EVAL_STEP = 2

#: The cycle sub-state machine, in forward order.
PHASE_STATES = ("started", "tokenizer_done", "refresh_done", "recommender_done")

#: The step a resumed cycle must re-enter, per recorded state.
STATE_ACTION = {
    "started": "tokenizer",
    "tokenizer_done": "refresh",
    "refresh_done": "recommender",
    "recommender_done": "next_cycle",
}

DEFAULT_SEED = 2025


class AlternatingError(RuntimeError):
    """Base class for E9 failures."""


class ScheduleError(AlternatingError, ValueError):
    """An impossible phase schedule was requested."""


class ResumeError(AlternatingError):
    """A cycle manifest is missing, malformed, or contradicts the run config."""


class HandoffError(AlternatingError):
    """The recommender did not consume the SID this cycle's refresh produced."""


class RefreshAborted(AlternatingError):
    """The SID refresh failed; the cycle stops before the recommender."""


# --------------------------------------------------------------------------- #
# Pure schedule logic
# --------------------------------------------------------------------------- #
def phase_of(epoch_index: int, cycle_length: int = CYCLE_LENGTH) -> str:
    """The stage of a global 0-based epoch index (``trainer.py:515``, verbatim).

    ``epoch_idx % cycle == 0`` -> tokenizer, else recommender. Note this is a
    modulo test on a **0-based** index: applying it to a 1-based label swaps
    every phase.
    """
    if cycle_length < 1:
        raise ScheduleError(f"cycle_length must be >= 1, got {cycle_length}")
    if epoch_index < 0:
        raise ScheduleError(f"epoch_index must be >= 0, got {epoch_index}")
    return STAGE_TOKENIZER if epoch_index % cycle_length == 0 else STAGE_RECOMMENDER


def cycle_of(epoch_index: int, cycle_length: int = CYCLE_LENGTH) -> int:
    """Which cycle a global 0-based epoch index belongs to."""
    if epoch_index < 0:
        raise ScheduleError(f"epoch_index must be >= 0, got {epoch_index}")
    return epoch_index // cycle_length


def epoch_index_of(cycle_index: int, stage: str,
                   cycle_length: int = CYCLE_LENGTH) -> int:
    """The global 0-based epoch index of ``stage`` inside ``cycle_index``.

    Only defined for the E9 schedule -- one tokenizer epoch followed by one
    recommender epoch per cycle -- so ``cycle_length`` must be 2 and the two
    stages map to offsets 0 and 1.
    """
    if cycle_length != CYCLE_LENGTH:
        raise ScheduleError(
            f"E9 defines a cycle as exactly {CYCLE_LENGTH} epochs "
            f"(1 tokenizer + 1 recommender); got cycle_length={cycle_length}")
    if stage not in STAGES:
        raise ScheduleError(f"unknown stage {stage!r}; expected one of {STAGES}")
    if cycle_index < 0:
        raise ScheduleError(f"cycle_index must be >= 0, got {cycle_index}")
    offset = 0 if stage == STAGE_TOKENIZER else 1
    return cycle_index * cycle_length + offset


def phase_plan(n_cycles: int, cycle_length: int = CYCLE_LENGTH) -> list:
    """The full schedule, as ``[{"cycle", "epoch", "stage"}, ...]``.

    The initial SID refresh is deliberately absent: it belongs to no cycle.
    """
    if n_cycles < 0:
        raise ScheduleError(f"n_cycles must be >= 0, got {n_cycles}")
    return [{"cycle": c, "epoch": e, "stage": phase_of(e, cycle_length)}
            for c in range(n_cycles)
            for e in range(c * cycle_length, (c + 1) * cycle_length)]


def check_eval_cadence(cycle_length: int = CYCLE_LENGTH,
                       eval_step: int = EVAL_STEP) -> None:
    """``trainer.py:56``'s invariant, kept so evaluation lands after a recommender.

    With ``cycle_length = eval_step = 2`` every evaluation falls on an odd
    (recommender) epoch. A cadence that does not divide the period would put
    evaluations after tokenizer epochs, which the E9 spec forbids.
    """
    if eval_step < 1:
        raise ScheduleError(f"eval_step must be >= 1, got {eval_step}")
    if cycle_length % eval_step != 0:
        raise ScheduleError(
            f"cycle_length ({cycle_length}) must be divisible by eval_step "
            f"({eval_step}); ETEGRec asserts this at trainer.py:56 because a "
            f"non-dividing cadence would evaluate after a tokenizer epoch")


def phase_seed(base_seed: int, epoch_index: int) -> int:
    """The seed of a phase: ``base_seed + global epoch index``.

    Deterministic and injective in the epoch index, so consecutive phases do not
    silently share a stream. The initial refresh (belonging to no epoch) uses
    ``base_seed`` itself.
    """
    return int(base_seed) + int(epoch_index)


def render_plan(plan) -> str:
    """A one-line-per-epoch rendering of a schedule, for logs and reports."""
    return "\n".join(
        f"    cycle {p['cycle']}  epoch {p['epoch']:>3}  {p['stage']}" for p in plan)


# --------------------------------------------------------------------------- #
# The cycle manifest
# --------------------------------------------------------------------------- #
def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def cycle_manifest_path(run_dir: str, cycle_index: int) -> str:
    return os.path.join(run_dir, "cycles", f"cycle_{int(cycle_index):03d}.json")


class CycleManifest:
    """The on-disk state machine of one cycle (E9 acceptance ②/③).

    Records, per cycle: the two epoch indices, the tokenizer checkpoint, the SID
    version, the recommender checkpoint, the evaluation, and every state
    transition with its timestamp. Written atomically (temp + ``os.replace``),
    the same convention E8 uses for its SID commit.

    Transitions are forward-only: :meth:`advance` refuses to skip a state or to
    move backwards, so a partially written manifest can never claim more
    progress than the run actually made.
    """

    SCHEMA = "tiger_ext.alternating/cycle-manifest/1"

    def __init__(self, path, record):
        self.path = path
        self.record = record

    # ------------------------------------------------------------ construction
    @classmethod
    def create(cls, run_dir, cycle_index, dataset, cycle_length=CYCLE_LENGTH,
               seed=DEFAULT_SEED, device=None):
        path = cycle_manifest_path(run_dir, cycle_index)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(path):
            raise ResumeError(f"a cycle manifest already exists at {path}")
        now = _now()
        record = {
            "schema": cls.SCHEMA,
            "dataset": dataset,
            "cycle_index": int(cycle_index),
            "cycle_length": int(cycle_length),
            "epoch_index": {
                STAGE_TOKENIZER: epoch_index_of(cycle_index, STAGE_TOKENIZER, cycle_length),
                STAGE_RECOMMENDER: epoch_index_of(cycle_index, STAGE_RECOMMENDER, cycle_length),
            },
            "stage_order": [STAGE_TOKENIZER, STAGE_RECOMMENDER],
            "state": PHASE_STATES[0],
            "seed": int(seed),
            "device": None if device is None else str(device),
            "created_at": now,
            "updated_at": now,
            "history": [{"state": PHASE_STATES[0], "at": now}],
            "tokenizer": None,
            "sid": None,
            "recommender": None,
            "evaluation": None,
            "errors": [],
        }
        manifest = cls(path, record)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            record = json.load(fh)
        if record.get("schema") != cls.SCHEMA:
            raise ResumeError(f"{path}: unknown manifest schema {record.get('schema')!r}")
        if record.get("state") not in PHASE_STATES:
            raise ResumeError(f"{path}: unknown state {record.get('state')!r}")
        return cls(path, record)

    @classmethod
    def discover(cls, run_dir) -> dict:
        """Every cycle manifest under ``run_dir``, keyed by cycle index."""
        directory = os.path.join(run_dir, "cycles")
        if not os.path.isdir(directory):
            return {}
        out = {}
        for name in sorted(os.listdir(directory)):
            if not (name.startswith("cycle_") and name.endswith(".json")):
                continue
            manifest = cls.load(os.path.join(directory, name))
            out[int(manifest.record["cycle_index"])] = manifest
        return out

    # --------------------------------------------------------------- accessors
    @property
    def cycle_index(self) -> int:
        return int(self.record["cycle_index"])

    @property
    def state(self) -> str:
        return self.record["state"]

    @property
    def state_index(self) -> int:
        return PHASE_STATES.index(self.record["state"])

    def get(self, key, default=None):
        return self.record.get(key, default)

    def epoch_of(self, stage: str) -> int:
        return int(self.record["epoch_index"][stage])

    def next_action(self) -> str:
        """Which step a resume from this state must re-enter."""
        return STATE_ACTION[self.record["state"]]

    # --------------------------------------------------------------- mutation
    def advance(self, state: str, **sections) -> "CycleManifest":
        """Move to the next state, recording the evidence for it.

        ``sections`` may carry ``tokenizer`` / ``sid`` / ``recommender``. A skip
        or a backwards move raises instead of being written.
        """
        if state not in PHASE_STATES:
            raise ResumeError(f"unknown state {state!r}")
        if PHASE_STATES.index(state) != self.state_index + 1:
            nxt = PHASE_STATES[self.state_index + 1] \
                if self.state_index + 1 < len(PHASE_STATES) else None
            raise ResumeError(
                f"cycle {self.cycle_index}: cannot advance from "
                f"{self.record['state']!r} to {state!r}; "
                + (f"the next state is {nxt!r}"
                   if nxt is not None else
                   f"{self.record['state']!r} is the terminal state of a cycle"))
        for key, value in sections.items():
            if key not in ("tokenizer", "sid", "recommender"):
                raise ResumeError(f"unknown manifest section {key!r}")
            self.record[key] = value
        self._touch(state)
        return self

    def record_evaluation(self, evaluation: dict) -> "CycleManifest":
        """Attach the post-recommender evaluation (no state change)."""
        if self.record["state"] != "recommender_done":
            raise ResumeError(
                f"cycle {self.cycle_index}: evaluation is only allowed after "
                f"recommender_done, state is {self.record['state']!r}")
        self.record["evaluation"] = dict(evaluation)
        self.record["updated_at"] = _now()
        self.save()
        return self

    def note_error(self, kind: str, message: str) -> "CycleManifest":
        """Record a failure without moving the state forward."""
        self.record["errors"].append(
            {"kind": str(kind), "message": str(message), "at": _now(),
             "state": self.record["state"]})
        self.record["updated_at"] = _now()
        self.save()
        return self

    def _touch(self, state: str) -> None:
        self.record["state"] = state
        self.record["updated_at"] = _now()
        self.record["history"].append({"state": state, "at": self.record["updated_at"]})
        self.save()

    # ------------------------------------------------------------------- io
    def save(self) -> str:
        tmp = f"{self.path}.tmp-{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(self.record, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, self.path)          # atomic: a reader never sees a half file
        return self.path

    def to_dict(self) -> dict:
        return dict(self.record)


# --------------------------------------------------------------------------- #
# The scheduler
# --------------------------------------------------------------------------- #
class AlternatingScheduler:
    """Drives ``initial refresh -> (T -> refresh -> R) x n_cycles``.

    The four phase bodies are injected, so the loop, the state machine, the
    failure policy and the hand-off check are testable without any training:

        tokenizer_phase(cycle_index, epoch_index, resume_ckpt)   -> dict
        refresh_phase(cycle_index, epoch_index, tokenizer_ckpt)  -> dict
        recommender_phase(cycle_index, epoch_index, sid_path,
                          sid_version, resume_ckpt)              -> dict
        eval_phase(cycle_index, epoch_index, recommender_ckpt,
                   sid_path)                                     -> dict

    Required keys of each return value:

        tokenizer   : ``ckpt_path``, ``sha256``
        refresh     : ``version``, ``sid_path``, ``sid_sha256``
        recommender : ``ckpt_path``, ``sha256``, ``sid_path_used``,
                      ``sid_version_used``, ``sid_sha256_used``
        eval        : free-form; recorded verbatim

    Args:
        run_dir: where the manifests are written. Owned by the caller.
        dataset: dataset name (also the SID file-name prefix, see E8).
        n_cycles: how many complete alternations to run.
        initial_tokenizer_ckpt: the tokenizer the first cycle starts from.
        cycle_length / eval_step: see :func:`phase_of` / :func:`check_eval_cadence`.
        seed: base seed; each phase uses ``phase_seed(seed, epoch_index)``.
        device: recorded in the manifest; the phase callables own the real device.
    """

    def __init__(self, run_dir, dataset, n_cycles, tokenizer_phase, refresh_phase,
                 recommender_phase, eval_phase=None, cycle_length=CYCLE_LENGTH,
                 eval_step=EVAL_STEP, seed=DEFAULT_SEED, device=None,
                 initial_tokenizer_ckpt=None, initial_recommender_ckpt=None,
                 initial_refresh=True):
        if cycle_length != CYCLE_LENGTH:
            raise ScheduleError(
                f"E9 defines a cycle as exactly {CYCLE_LENGTH} epochs "
                f"(1 tokenizer + 1 recommender); got cycle_length={cycle_length}")
        check_eval_cadence(cycle_length, eval_step)
        if n_cycles < 0:
            raise ScheduleError(f"n_cycles must be >= 0, got {n_cycles}")
        self.run_dir = run_dir
        self.dataset = dataset
        self.n_cycles = int(n_cycles)
        self.cycle_length = int(cycle_length)
        self.eval_step = int(eval_step)
        self.seed = int(seed)
        self.device = None if device is None else str(device)
        self.tokenizer_phase = tokenizer_phase
        self.refresh_phase = refresh_phase
        self.recommender_phase = recommender_phase
        self.eval_phase = eval_phase
        self.initial_tokenizer_ckpt = initial_tokenizer_ckpt
        self.initial_recommender_ckpt = initial_recommender_ckpt
        self.initial_refresh_enabled = bool(initial_refresh)
        self.initial_manifest_path = os.path.join(run_dir, "initial_refresh.json")

    # ------------------------------------------------------------------ plan
    @property
    def plan(self) -> list:
        return phase_plan(self.n_cycles, self.cycle_length)

    def total_epochs(self) -> int:
        return self.n_cycles * self.cycle_length

    # -------------------------------------------------------- initial refresh
    def initial_refresh_record(self):
        """The initial refresh, already done -- or ``None``."""
        if os.path.isfile(self.initial_manifest_path):
            with open(self.initial_manifest_path) as fh:
                return json.load(fh)
        return None

    def run_initial_refresh(self) -> dict:
        """Refresh the SID from the starting tokenizer, outside any cycle.

        Idempotent: an already-written record is returned unchanged, so a resume
        never burns a second version number here.
        """
        existing = self.initial_refresh_record()
        if existing is not None:
            return existing
        if not self.initial_refresh_enabled:
            return {"skipped": True, "reason": "initial_refresh disabled"}
        if not self.initial_tokenizer_ckpt:
            raise ScheduleError(
                "no initial tokenizer checkpoint; the initial SID refresh "
                "cannot run (pass initial_tokenizer_ckpt=...)")
        record = dict(self.refresh_phase(-1, -1, self.initial_tokenizer_ckpt))
        record["_phase"] = "initial_refresh"
        os.makedirs(self.run_dir, exist_ok=True)
        tmp = f"{self.initial_manifest_path}.tmp-{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, self.initial_manifest_path)
        return record

    # ------------------------------------------------------------------- run
    def run(self):
        """Run (or resume) the whole schedule; returns the per-cycle records."""
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_initial_refresh()

        existing = CycleManifest.discover(self.run_dir)
        complete = [c for c in sorted(existing) if existing[c].state == "recommender_done"
                    and existing[c].get("evaluation") is not None]
        # resume the first cycle that is not fully complete; anything after it is
        # ignored (it cannot exist, since cycles run strictly in order)
        first_open = next((c for c in range(self.n_cycles) if c not in complete), None)

        prev_tok = self._previous_tokenizer(first_open, existing)
        prev_rec = self._previous_recommender(first_open, existing)
        if first_open is None:
            return [existing[c].to_dict() for c in sorted(existing)]

        for cycle_index in range(first_open, self.n_cycles):
            manifest = existing.get(cycle_index)
            if manifest is None:
                manifest = CycleManifest.create(
                    self.run_dir, cycle_index, self.dataset,
                    cycle_length=self.cycle_length, seed=self.seed, device=self.device)
            prev_tok, prev_rec = self.run_cycle(cycle_index, manifest, prev_tok, prev_rec)
        return self.summary()

    def _previous_tokenizer(self, first_open, existing):
        if first_open:
            prior = existing.get(first_open - 1) or existing.get(max(existing))
            if prior is not None and prior.get("tokenizer"):
                return prior.get("tokenizer")["ckpt_path"]
        initial = self.initial_refresh_record()
        if initial and initial.get("ckpt_path"):
            return initial["ckpt_path"]
        return self.initial_tokenizer_ckpt

    def _previous_recommender(self, first_open, existing):
        if first_open:
            prior = existing.get(first_open - 1)
            if prior is not None and prior.get("recommender"):
                return prior.get("recommender")["ckpt_path"]
        return self.initial_recommender_ckpt

    def run_cycle(self, cycle_index, manifest, prev_tok, prev_rec):
        """Drive one cycle from its recorded state to ``recommender_done``.

        Each branch is guarded by the recorded state, which is what makes the
        resume idempotent: entering at ``refresh_done`` never re-runs the
        refresh, and entering at ``tokenizer_done`` never re-runs the tokenizer.
        """
        action = manifest.next_action()
        epoch_tok = manifest.epoch_of(STAGE_TOKENIZER)
        epoch_rec = manifest.epoch_of(STAGE_RECOMMENDER)

        if action == "tokenizer":
            result = dict(self.tokenizer_phase(cycle_index, epoch_tok, prev_tok))
            _require(result, ("ckpt_path", "sha256"), "tokenizer_phase")
            manifest.advance("tokenizer_done", tokenizer=result)
            prev_tok = result["ckpt_path"]
            action = manifest.next_action()

        if action == "refresh":
            ckpt = manifest.get("tokenizer")["ckpt_path"]
            _verify_unchanged(ckpt, manifest.get("tokenizer")["sha256"],
                              f"cycle {cycle_index} tokenizer checkpoint")
            try:
                refreshed = dict(self.refresh_phase(cycle_index, epoch_tok, ckpt))
            except BaseException as exc:
                # Failure policy: record, stop the cycle, never fall through to
                # the recommender (E9 spec §5 "Refresh 失败 -> stop").
                manifest.note_error(type(exc).__name__, str(exc))
                raise RefreshAborted(
                    f"cycle {cycle_index}: the SID refresh failed, so the "
                    f"recommender phase must not run ({type(exc).__name__}: {exc})"
                ) from exc
            _require(refreshed, ("version", "sid_path", "sid_sha256"), "refresh_phase")
            manifest.advance("refresh_done", sid=refreshed)
            action = manifest.next_action()

        if action == "recommender":
            sid = manifest.get("sid")
            _verify_unchanged(sid["sid_path"], sid["sid_sha256"],
                              f"cycle {cycle_index} SID")
            result = dict(self.recommender_phase(
                cycle_index, epoch_rec, sid["sid_path"], sid["version"], prev_rec))
            _require(result, ("ckpt_path", "sha256", "sid_path_used",
                              "sid_version_used", "sid_sha256_used"),
                     "recommender_phase")
            self.check_handoff(cycle_index, sid, result)
            manifest.advance("recommender_done", recommender=result)
            prev_rec = result["ckpt_path"]
            action = manifest.next_action()

        if action == "next_cycle" and self.eval_phase is not None \
                and manifest.get("evaluation") is None:
            sid = manifest.get("sid")
            evaluation = dict(self.eval_phase(cycle_index, epoch_rec,
                                              manifest.get("recommender")["ckpt_path"],
                                              sid["sid_path"]))
            manifest.record_evaluation(evaluation)
        return prev_tok, prev_rec

    @staticmethod
    def check_handoff(cycle_index, sid_record, recommender_result) -> bool:
        """The SID the recommender actually read must be this cycle's refresh.

        Compares the version and the content hash of the file the recommender
        reports having consumed against the manifest entry the refresh wrote.
        A recommender that fell back to the previous cycle's SID -- or to the
        canonical one -- fails here instead of being recorded as a success.
        """
        problems = []
        if recommender_result["sid_version_used"] != sid_record["version"]:
            problems.append(
                f"SID version {recommender_result['sid_version_used']} != refreshed "
                f"version {sid_record['version']}")
        if recommender_result["sid_sha256_used"] != sid_record["sid_sha256"]:
            problems.append(
                f"SID sha256 {str(recommender_result['sid_sha256_used'])[:12]} != "
                f"refreshed sha256 {str(sid_record['sid_sha256'])[:12]}")
        if os.path.abspath(recommender_result["sid_path_used"]) != \
                os.path.abspath(sid_record["sid_path"]):
            problems.append(
                f"SID path {recommender_result['sid_path_used']} != refreshed "
                f"path {sid_record['sid_path']}")
        if problems:
            raise HandoffError(
                f"cycle {cycle_index}: the recommender did not consume the SID this "
                f"cycle's refresh produced:\n  - " + "\n  - ".join(problems))
        return True

    # --------------------------------------------------------------- reporting
    def summary(self):
        """Every cycle manifest, ordered -- the run's audit chain (验收 ②)."""
        return [m.to_dict() for _, m in sorted(CycleManifest.discover(self.run_dir).items())]

    def resume_plan(self) -> dict:
        """What a fresh ``run()`` would do right now (used by tests and reports)."""
        existing = CycleManifest.discover(self.run_dir)
        out = {"initial_refresh": self.initial_refresh_record() is not None,
               "actions": []}
        for cycle_index in range(self.n_cycles):
            manifest = existing.get(cycle_index)
            if manifest is None:
                out["actions"].append({"cycle": cycle_index, "action": "tokenizer",
                                       "state": None})
                continue
            action = manifest.next_action()
            if action == "next_cycle" and manifest.get("evaluation") is None \
                    and self.eval_phase is not None:
                action = "evaluate"
            out["actions"].append({"cycle": cycle_index, "action": action,
                                   "state": manifest.state})
        return out


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def _require(mapping, keys, what):
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ResumeError(f"{what} did not return {missing}; got {sorted(mapping)}")


def _verify_unchanged(path, expected_sha, what):
    """Refuse to resume off an artifact that changed since it was registered."""
    if not os.path.isfile(path):
        raise ResumeError(f"{what} is gone: {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ResumeError(
            f"{what} changed since it was recorded ({str(expected_sha)[:12]} -> "
            f"{actual[:12]}); refusing to resume off a different artifact")


__all__ = [
    "AlternatingError", "ScheduleError", "ResumeError", "HandoffError",
    "RefreshAborted", "CycleManifest", "AlternatingScheduler",
    "phase_of", "cycle_of", "epoch_index_of", "phase_plan", "phase_seed",
    "check_eval_cadence", "render_plan", "cycle_manifest_path",
    "PHASE_STATES", "STATE_ACTION", "CYCLE_LENGTH", "EVAL_STEP", "DEFAULT_SEED",
]
