"""Ported modules for the CSG4Rec / ETEGRec -> TIGER integration.

Held outside ``model/`` and ``rqvae/`` on purpose: those two directories are the
frozen Clean TIGER core and the frozen RQ-VAE, and the porting plan requires the
new mechanisms to live beside them rather than inside them.

Each module here is a self-contained building block. The SIA / PSA blocks are
mounted into the TIGER loop by the caller (E4 / E6); the stage layer (E7) only
flips ``requires_grad`` and reports which losses belong to a stage -- it never
touches either training loop. The SID refresh layer (E8) drives the frozen
``rqvae/generate_code.py`` through its existing CLI and adds versioned naming
plus a provenance manifest -- it adds no encoding mechanism of its own. The CSG
line starts at C1 (``cf_embed.py``): the item-level collaborative-embedding
interface, a numpy-only leaf that computes no representation of its own. C2
(``collab_relation.py``) turns that table into a directed top-k cosine
item-to-item relation -- likewise a numpy-only leaf, and likewise a side artifact:
it feeds no one's input. C3 (``qspace_constraint.py``) projects that relation onto
the RQ-VAE's quantised representation: it emits the per-batch constraint
intermediate, and deliberately computes no loss -- the objective is C4's. It reads
the relation only through C2's own ``CollabRelation.load``, and it captures the
Q-space tensor only from the model's forward pass, checking on the spot that what
it captured really is the codebook sum rather than the encoder latent. C4
(``qspace_loss.py``) is the other half: it turns that intermediate into the
scalar objective -- a per-level cross-entropy of the anchor's codeword against
the level's own codebook, with the collaborative neighbour's codeword as the
class -- proving in its own harness that the gradient arrives at the quantiser
and that the C2 weight, which the reference never uses, is carried but unread.
It is standalone: no trainer, no stage switch, no optimiser. Wiring it in is C5's
task. Note the C4 constants are namespaced ``QSL_*``: several names below
(``MISSING_REFERENCE``, ``DIRECTION_POLICY``) already belong to C2 and C3, and a
second export of the same name would silently shadow the first.
"""
from .sia import SIAAdapter, masked_mean_pool, SIA_LATENT_DIM
from .sia_codes import SIACodeExtractor
from .sia_loss import SIAMount, SIALoss, symmetric_kl
from .psa import PSAAdapter, psa_position0, PSA_SEQ_POS
from .psa_loss import PSAMount, PSALoss, symmetric_contrastive_loss
from .stage import (StageFreeze, StageError, StageInconsistent, StageMismatch,
                    STAGES, STAGE_TOKENIZER, STAGE_RECOMMENDER, STAGE_CONTRACT,
                    GROUP_TOKENIZER, GROUP_RECOMMENDER, GROUP_ADAPTER,
                    codebook_layers, codebook_initted, codebook_is_zero,
                    seal_codebooks, load_tokenizer_state_dict,
                    stage_manifest_path, write_stage_manifest,
                    read_stage_manifest, verify_stage_manifest)
from .cf_embed import (CFEmbedding, CFEmbeddingError, CFEmbeddingMismatch,
                       MissingItemError, canonical_item_ids, default_cf_path,
                       item_ids_sha256, load_cf_embedding,
                       load_sasrec_item_embedding, sasrec_checkpoint_path,
                       CF_DTYPE, PARQUET_COLUMNS,
                       CF_SOURCE_SASREC, CF_SOURCE_ITEMCF, DEFAULT_CF_SOURCE,
                       SASREC_OUTPUT_DIR, SASREC_CKPT_TEMPLATE, SASREC_EMB_KEY,
                       SASREC_PADDING_ROW, SASREC_DIM, SASREC_ROW_OF_ITEM,
                       SASREC_PADDING_HANDLING, SASREC_COLD_START, SASREC_RECIPE,
                       ITEMCF_COLD_START,
                       DEFAULT_CF_TEMPLATE, CF_SOURCE_VARIANT, CF_SOURCE_RECIPE,
                       CF_SOURCE_GENERATOR)
from .collab_relation import (CollabRelation, CollabRelationError,
                              CollabRelationMismatch, build_relation,
                              build_relation_for_dataset, l2_normalize,
                              topk_cosine_neighbors, relation_prefix,
                              RELATION_K, RELATION_METRIC, RELATION_DTYPE,
                              SIMILARITY_DTYPE, RELATION_SPACE, SIM_FORMULA,
                              SELF_EXCLUSION, ZERO_NORM_POLICY, DIRECTEDNESS,
                              TIE_BREAK, PAIR_FILE_NOTE, REFERENCE_IMPLEMENTATION,
                              PORT_DIFFERENCES, ARTIFACT_DIRNAME,
                              DEFAULT_BLOCK, SUFFIX)
from .qspace_constraint import (QSpaceConstraint, QSpaceBatch, ConstraintBatch,
                                QSpaceError, QSpaceMismatch, QSpaceDisabled,
                                capture_qspace, rebuild_quantized,
                                load_constraint_for_dataset, rows_of_item_ids,
                                QSPACE_REPRESENTATION, QSPACE_EXCLUDED,
                                QSPACE_SOURCE, QSPACE_PROVENANCE_CHECK,
                                THREE_PARTS, BATCH_POLICY, SENTINEL_POLICY,
                                DIRECTION_POLICY, REFERENCE_IMPLEMENTATION,
                                MISSING_REFERENCE, PORT_DIFFERENCES,
                                DEFAULT_LAMBDA_CSG, REBUILD_ATOL, REBUILD_RTOL)
from .qspace_loss import (QSpaceLoss, QSpaceLossMount, QSpaceLossError,
                          qspace_constraint_loss, codebook_stack,
                          verify_indices_match_qspace,
                          QSL_OBJECTIVE_FORM, QSL_REFERENCE_OBJECTIVE,
                          QSL_MISSING_REFERENCE, QSL_DESIGN_DECISIONS,
                          QSL_WEIGHT_POLICY, QSL_SELF_MASK, QSL_SELF_MASK_RULE,
                          QSL_SAME_CODE_POLICY, QSL_LEVEL_POLICY,
                          QSL_MULTI_NEIGHBOUR_POLICY, QSL_DIRECTION_POLICY,
                          QSL_EMPTY_POLICY, QSL_GRADIENT_POLICY,
                          QSL_REFERENCE_SCALE)
from .qspace_train import (CSGHook, CSGTrainerConfig, QSpaceTrainError,
                           train_tokenizer, build_model, build_dataset,
                           frozen_tokenizer_modules, build_parser,
                           CST_TOTAL_LOSS_FORM, CST_GRADIENT_ROUTES,
                           CST_GRADIENT_POLICY, CST_LAMBDA_CANDIDATES,
                           CST_LAMBDA_POLICY, CST_REFERENCE_WEIGHT,
                           CST_SWITCH_CONTRACT, CST_STAGE_NOTE, CST_ENTRY_NOTE,
                           CST_PENDING_DECISIONS, CST_DEFAULT_SEED,
                           CST_SANITY_SUBSET)
from .qspace_smoke import (SmokeSpec, QSpaceSmokeError, synthetic_spec,
                           toys_spec, build_fixture, build_hook, pinned_loader,
                           run_chain, probe_kmeans, rebuild_delta_of,
                           CST_SMOKE_LAMBDAS, CST_SMOKE_K, CST_SMOKE_ARM_NOTE,
                           CST_SMOKE_SYN, CST_SMOKE_SYN_RQVAE, CST_SMOKE_SYN_T5,
                           CST_SMOKE_TOYS, CST_SMOKE_TOYS_NOTE,
                           CST_SMOKE_PENDING)
from .sid_refresh import (refresh_sid, run_generator, publish_sid,
                          validate_sid, collision_stats, diff_rate, load_sid,
                          ckpt_provenance, n_items_of, next_version,
                          list_versions, parse_sid_name, sid_path, sid_name,
                          version_dir, sidecar_path, write_sid_manifest,
                          read_sid_manifest, verify_sid_manifest,
                          SidRefreshError, SidValidationError, SidManifestMismatch,
                          SID_COLUMNS, CODEBOOK_SIZE, GENERATOR_REL, sha256_file)
from .alternating import (AlternatingScheduler, CycleManifest, AlternatingError,
                          ScheduleError, ResumeError, HandoffError, RefreshAborted,
                          phase_of, cycle_of, epoch_index_of, phase_plan, phase_seed,
                          check_eval_cadence, render_plan, cycle_manifest_path,
                          PHASE_STATES, STATE_ACTION, CYCLE_LENGTH, EVAL_STEP,
                          DEFAULT_SEED)

__all__ = ["SIAAdapter", "masked_mean_pool", "SIA_LATENT_DIM", "SIACodeExtractor",
           "SIAMount", "SIALoss", "symmetric_kl",
           "PSAAdapter", "psa_position0", "PSA_SEQ_POS",
           "PSAMount", "PSALoss", "symmetric_contrastive_loss",
           "StageFreeze", "StageError", "StageInconsistent", "StageMismatch",
           "STAGES", "STAGE_TOKENIZER", "STAGE_RECOMMENDER", "STAGE_CONTRACT",
           "GROUP_TOKENIZER", "GROUP_RECOMMENDER", "GROUP_ADAPTER",
           "codebook_layers", "codebook_initted", "codebook_is_zero",
           "seal_codebooks", "load_tokenizer_state_dict",
           "stage_manifest_path", "write_stage_manifest",
           "read_stage_manifest", "verify_stage_manifest",
           "CFEmbedding", "CFEmbeddingError", "CFEmbeddingMismatch",
           "MissingItemError", "canonical_item_ids", "default_cf_path",
           "item_ids_sha256", "load_cf_embedding", "load_sasrec_item_embedding",
           "sasrec_checkpoint_path", "CF_DTYPE", "PARQUET_COLUMNS",
           "CF_SOURCE_SASREC", "CF_SOURCE_ITEMCF", "DEFAULT_CF_SOURCE",
           "SASREC_OUTPUT_DIR", "SASREC_CKPT_TEMPLATE", "SASREC_EMB_KEY",
           "SASREC_PADDING_ROW", "SASREC_DIM", "SASREC_ROW_OF_ITEM",
           "SASREC_PADDING_HANDLING", "SASREC_COLD_START", "SASREC_RECIPE",
           "ITEMCF_COLD_START",
           "DEFAULT_CF_TEMPLATE", "CF_SOURCE_VARIANT", "CF_SOURCE_RECIPE",
           "CF_SOURCE_GENERATOR",
           "CollabRelation", "CollabRelationError", "CollabRelationMismatch",
           "build_relation", "build_relation_for_dataset", "l2_normalize",
           "topk_cosine_neighbors", "relation_prefix",
           "RELATION_K", "RELATION_METRIC", "RELATION_DTYPE", "SIMILARITY_DTYPE",
           "RELATION_SPACE", "SIM_FORMULA", "SELF_EXCLUSION", "ZERO_NORM_POLICY",
           "DIRECTEDNESS", "TIE_BREAK", "PAIR_FILE_NOTE", "REFERENCE_IMPLEMENTATION",
           "PORT_DIFFERENCES", "ARTIFACT_DIRNAME", "DEFAULT_BLOCK", "SUFFIX",
           "refresh_sid", "run_generator", "publish_sid",
           "validate_sid", "collision_stats", "diff_rate", "load_sid",
           "ckpt_provenance", "n_items_of", "next_version",
           "list_versions", "parse_sid_name", "sid_path", "sid_name",
           "version_dir", "sidecar_path", "write_sid_manifest",
           "read_sid_manifest", "verify_sid_manifest",
           "SidRefreshError", "SidValidationError", "SidManifestMismatch",
           "SID_COLUMNS", "CODEBOOK_SIZE", "GENERATOR_REL", "sha256_file",
           "QSpaceConstraint", "QSpaceBatch", "ConstraintBatch",
           "QSpaceError", "QSpaceMismatch", "QSpaceDisabled",
           "capture_qspace", "rebuild_quantized", "load_constraint_for_dataset",
           "rows_of_item_ids",
           "QSPACE_REPRESENTATION", "QSPACE_EXCLUDED", "QSPACE_SOURCE",
           "QSPACE_PROVENANCE_CHECK", "THREE_PARTS", "BATCH_POLICY",
           "SENTINEL_POLICY", "DIRECTION_POLICY", "REFERENCE_IMPLEMENTATION",
           "MISSING_REFERENCE", "PORT_DIFFERENCES", "DEFAULT_LAMBDA_CSG",
           "REBUILD_ATOL", "REBUILD_RTOL",
           "QSpaceLoss", "QSpaceLossMount", "QSpaceLossError",
           "qspace_constraint_loss", "codebook_stack", "verify_indices_match_qspace",
           "QSL_OBJECTIVE_FORM", "QSL_REFERENCE_OBJECTIVE",
           "QSL_MISSING_REFERENCE", "QSL_DESIGN_DECISIONS",
           "QSL_WEIGHT_POLICY", "QSL_SELF_MASK", "QSL_SELF_MASK_RULE",
           "QSL_SAME_CODE_POLICY", "QSL_LEVEL_POLICY",
           "QSL_MULTI_NEIGHBOUR_POLICY", "QSL_DIRECTION_POLICY",
           "QSL_EMPTY_POLICY", "QSL_GRADIENT_POLICY", "QSL_REFERENCE_SCALE",
           "CSGHook", "CSGTrainerConfig", "QSpaceTrainError",
           "train_tokenizer", "build_model", "build_dataset",
           "frozen_tokenizer_modules", "build_parser",
           "CST_TOTAL_LOSS_FORM", "CST_GRADIENT_ROUTES", "CST_GRADIENT_POLICY",
           "CST_LAMBDA_CANDIDATES", "CST_LAMBDA_POLICY", "CST_REFERENCE_WEIGHT",
           "CST_SWITCH_CONTRACT", "CST_STAGE_NOTE", "CST_ENTRY_NOTE",
           "CST_PENDING_DECISIONS", "CST_DEFAULT_SEED", "CST_SANITY_SUBSET",
           "SmokeSpec", "QSpaceSmokeError", "synthetic_spec", "toys_spec",
           "build_fixture", "build_hook", "pinned_loader", "run_chain",
           "probe_kmeans", "rebuild_delta_of",
           "CST_SMOKE_LAMBDAS", "CST_SMOKE_K", "CST_SMOKE_ARM_NOTE",
           "CST_SMOKE_SYN", "CST_SMOKE_SYN_RQVAE", "CST_SMOKE_SYN_T5",
           "CST_SMOKE_TOYS", "CST_SMOKE_TOYS_NOTE", "CST_SMOKE_PENDING",
           "AlternatingScheduler", "CycleManifest", "AlternatingError",
           "ScheduleError", "ResumeError", "HandoffError", "RefreshAborted",
           "phase_of", "cycle_of", "epoch_index_of", "phase_plan", "phase_seed",
           "check_eval_cadence", "render_plan", "cycle_manifest_path",
           "PHASE_STATES", "STATE_ACTION", "CYCLE_LENGTH", "EVAL_STEP",
           "DEFAULT_SEED"]
