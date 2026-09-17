#!/usr/bin/env python3
"""Hybrid time-frequency / time-domain direct RIR estimation.

Answers the question the 2026-08-06 meeting left open: the previous direct-RIR
gate failed, but with an architecture whose inductive bias does not match a room
impulse response.  This runner re-tests direct supervised RIR estimation with the
hybrid backbone of Moliner et al. (Sec. III-B, Fig. 1), against the *same* data,
horizon, objective, optimiser and budget as the frozen WaveNet backbone, so that
architecture is the only variable.

Stages
------
``overfit``   Can the architecture fit a handful of training examples at all?
              This is a prerequisite: the previous gate failed here (exact-train
              NRMSE 1.447, worse than predicting zeros), so a held-out
              comparison would be uninterpretable without it.
``train``     Full supervised training on the frozen provider shards.
``evaluate``  Held-out comparison against the analytic baselines the meeting
              named: 3 ms truncation -> regularized deconvolution, and
              true-x regularized deconvolution.

Everything here is K=1.  Multiple claps are a second-stage question and are only
worth asking if single-clap estimation beats the deconvolution baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from claprir.metrics.deconvolution import (edc_rmse, regularized_deconvolution,
                                        rir_metrics)
from claprir.analysis.regularization_selection import (estimated_excitation,
                                                         load_estimator)
from claprir.models.hybrid_rir_estimator import (COMPRESSION_EXPONENT, HybridConfig,
                                       HybridDirectRIRFlow,
                                       HybridDirectRIRRegression, HybridEarlyFlow,
                                       HybridSetLate, HybridWeightedRegression,
                                       WaveNetDirectRIRRegression)

SAMPLE_RATE = 44_100
ARCHITECTURES = ("hybrid", "wavenet", "hybrid_early_flow", "hybrid_flow",
                 "hybrid_set_late", "hybrid_weighted")
#: Flow arms are stochastic; evaluation fixes these inference seeds and
#: reports the spread rather than a single draw.
INFERENCE_SEEDS = (0, 1, 2)
#: Checkpoint every 1000 updates.  These runs have twice been killed
#: mid-flight by something this environment does not let us identify, so
#: the schedule is set by how much work is acceptable to lose, not by how
#: many evaluation points are wanted.  ``prune_checkpoints`` keeps the
#: directory small.
MILESTONES = tuple(range(1_000, 20_001, 1_000)) + (500,)
DEFAULT_DATA_ROOT = Path("data/multiroom_generalization")
REPORT_ROOT = Path("reports/hybrid_direct_rir")
DEFAULT_RUN_ROOT = Path("runs/hybrid_direct_rir")


@dataclass(frozen=True)
class RunConfig:
    architecture: str
    training_datasets: tuple[str, ...]
    seed: int
    horizon_ms: int = 250
    updates: int = 20_000
    batch_size: int = 16
    learning_rate: float = 2e-4
    stft_weight: float = .25
    #: ``None`` trains on the whole split; an integer restricts training to that
    #: many fixed examples (the overfit sanity check).
    subset: int | None = None
    nf: int = 128
    #: Magnitude exponent for the 2-D stage's input spectrogram. 2/3 is the
    #: frozen default; 1.0 is the registered representation ablation
    #: (reports/representation_ablation). Carried on RunConfig so it lands in
    #: the checkpoint's provenance rather than being an untracked flag.
    input_compression: float = COMPRESSION_EXPONENT
    #: Step 4A: weight on the multi-resolution STFT term. 0.0 is the frozen
    #: default; see reports/spectral_loss_ladder/preregistration.md for why the
    #: value used there is 0.02 and why it is not swept.
    spectral_weight: float = 0.0
    #: Exponent inside the auxiliary spectral term (NOT the model input). 2/3 is
    #: frozen; 1.0 makes it the plain STFT-magnitude MSE of the original notes.
    spectral_exponent: float = COMPRESSION_EXPONENT
    #: Supervise only where the target is real. The shards are a fixed 1 s buffer
    #: and most sources are shorter, so past a record's last real sample the
    #: target is zero because the RECORDING ENDED, not because the room fell
    #: silent (reports/target_provenance_audit). Training on those zeros teaches
    #: the model that those rooms stop, which is not known to be true. Off by
    #: default: every arm before this trained without it and turning it on
    #: silently would make new runs incomparable to old ones.
    validity_masking: bool = False
    #: "peak" is the frozen default: the shards already carry per-RIR peak
    #: normalisation from manifest.py and nothing further is applied.
    #: "global_rms" divides every target by ONE scalar a = mean_i RMS(h_i),
    #: fitted on the training split alone, so the target distribution sits at
    #: unit RMS against the unit-Gaussian source the flow transports from
    #: (reports/flow_formulation_audit measured a 29.9x mismatch there).
    #: Predictions are multiplied back by a before any metric, so results stay
    #: in the same domain as every other arm and remain comparable.
    #: This is a coordinate fix for the transport, NOT a restoration of physical
    #: inter-room amplitude -- reports/amplitude_convention_audit shows that is
    #: not recoverable from this data.
    target_normalisation: str = "peak"
    #: "dataset_balanced" (frozen default) or "rt60_balanced" (Wave 4 ablation).
    sampling: str = "dataset_balanced"
    #: "stft" (frozen default) or "multires" (Wave 3 ablation).
    analysis: str = "stft"
    #: Observation channels the model consumes at once. 1 is every frozen arm
    #: in this programme; >1 is the joint multi-clap model of
    #: ``multiclap_joint``, which pads to k_max and passes a presence mask.
    k_max: int = 1
    #: Micro-batches accumulated before each optimiser step. The EFFECTIVE batch
    #: is batch_size * accumulate, and the W3-A horizon ablation relies on that:
    #: a 1 s target does not fit at batch 16, and changing the batch size for one
    #: arm would confound horizon with optimisation. Accumulation keeps the
    #: effective batch, the batch CONTENTS and the sampler stream identical
    #: across arms, because store.sample draws per example.
    accumulate: int = 1

    def __post_init__(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")

    @property
    def signal_length(self) -> int:
        return round(SAMPLE_RATE * self.horizon_ms / 1000)

    @property
    def run_name(self) -> str:
        mixture = "_".join(self.training_datasets)
        scope = "full" if self.subset is None else f"subset{self.subset}"
        tag = "" if self.sampling == "dataset_balanced" else f"_{self.sampling}"
        tag += "" if self.analysis == "stft" else f"_{self.analysis}"
        # k=1 keeps the frozen names byte-for-byte, so no existing checkpoint
        # path, archive entry or provenance record moves.
        tag += "" if self.k_max == 1 else f"_k{self.k_max}"
        # Every setting that changes what the weights mean has to be in the name,
        # for the reason the k_max line above already encodes. Two arms that
        # differ only in target_normalisation or input_compression used to
        # resolve to the SAME run_name, so the second would resume from the
        # first's checkpoint and then overwrite it -- silently, and looking like
        # a successful resume. The representation ablation worked around that by
        # hand, with separate run roots; a name that carries the setting removes
        # the need for the workaround and the chance of forgetting it.
        # Defaults contribute nothing, so no existing name changes.
        tag += "" if self.target_normalisation == "peak" \
            else f"_{self.target_normalisation}"
        tag += "" if self.input_compression == COMPRESSION_EXPONENT \
            else f"_c{self.input_compression:g}".replace(".", "p")
        # The Eloi-style pure-FM arm turns this off. It has to be in the name
        # for the reason input_compression above already gives: without it the
        # pure-FM arm and the C_plain arm resolve to the SAME directory, and the
        # pure-FM run would resume from C_plain's finished 20000-update
        # checkpoint and report itself complete without training a step.
        tag += "" if self.stft_weight == .25 \
            else f"_w{self.stft_weight:g}".replace(".", "p")
        # In the name for the reason the two exponents above already are: an arm
        # scored through a different spectral warp is a different experiment, and
        # two that shared a name would resume from each other's checkpoint.
        tag += "" if self.spectral_exponent == COMPRESSION_EXPONENT \
            else f"_x{self.spectral_exponent:g}".replace(".", "p")
        # In the name for the same reason as the exponents: an arm supervised on
        # a different set of samples is a different experiment.
        tag += "_vm" if self.validity_masking else ""
        tag += "" if not self.spectral_weight \
            else f"_s{self.spectral_weight:g}".replace(".", "p")
        return (f"{self.architecture}_{mixture}_{self.horizon_ms}ms_"
                f"{scope}{tag}_seed{self.seed}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_fingerprint() -> str:
    """Content hash of the clapgen package, independent of where it lives.

    A commit hash identifies code by label and needs a repository to resolve
    against. Staged jobs run from a *copy*, so there is no repository, and
    anchoring on ``__file__`` resolves to the copied ``src/`` -- which is how
    the first version of this fix still recorded nothing for exactly the runs it
    was written for. A content hash has no such dependency: it is computable
    from a staged tree, and it can be matched against ``git archive <commit>``
    afterwards to name the commit, which is how the retrained flow arms' commit
    was established.
    """
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit() -> str:
    """HEAD of the repository this module lives in, not of the process cwd.

    Every GPU job in this programme runs staged on local disk, which is not a
    git repository, so a bare ``git rev-parse HEAD`` returned an empty string
    and all 39 checkpoints recorded no training commit. Anchoring on ``__file__``
    is what makes the field survive staging. ``-C`` is used rather than a cwd
    change so this is safe to call from anywhere.

    Returns "" only if the source really is outside a repository, which is a
    fact worth recording rather than an error worth raising.
    """
    # CLAPGEN_REPO first: a staged job runs from a copy with no repository at
    # all, so the module path cannot resolve one and the launcher must say where
    # it is. The module path remains the fallback for in-repo use.
    for repository in (os.environ.get("CLAPGEN_REPO"),
                       str(Path(__file__).resolve().parents[3])):
        if not repository:
            continue
        result = subprocess.run(["git", "-C", repository, "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return ""


#: RT60 bucket edges in seconds, for the balanced-sampling ablation.
RT60_EDGES = (0.3, 0.6, 1.0)


class SingleClapStore:
    """Room-stratified sampler over the frozen one-second provider shards.

    Mirrors ``multiroom_generalization.direct_rir_estimation.ProviderStore`` --
    dataset chosen uniformly, then room, then record -- so that the hybrid model
    sees the same training distribution as the WaveNet gate it is replacing.
    Restricted to one observation per example, which is the K=1 question.

    ``sampling="rt60_balanced"`` adds a decay-time stratum between dataset and
    room. Dataset balancing alone does not prevent one acoustic regime from
    dominating: a provider with thousands of similar-RT60 configurations still
    contributes only its own reverberation times.

    The shards carry no RT60 field, so buckets are estimated -- from the **full
    stored RIR** (1 s), never from the 250 ms model target, because a target
    crop cannot distinguish a 0.6 s room from a 1.2 s one. ``decay_estimate``
    supplies the fit interval, R^2 and censored/extrapolated flags with each
    value, and ``bucket_manifest`` exposes them so the stratification can be
    audited rather than trusted.

    A censored record -- one whose decay outlasts even the 1 s shard -- goes in
    the LONGEST bucket. An earlier version sent unmeasurable decays to the
    shortest one, which is backwards: failing to reach -35 dB is evidence of a
    long decay, so that choice biased the stratification against exactly the
    rooms it was meant to surface.
    """

    #: Providers whose zeros are a simulator's finite support rather than a
    #: recording that stopped. Their post-support samples are KNOWN to be zero,
    #: so they are supervised for the whole horizon. See
    #: reports/target_provenance_audit and docs/EDC_PROTOCOL.md.
    SYNTHETIC_PROVIDERS = ("shoebox",)

    def __init__(self, datasets: tuple[str, ...], split: str, length: int,
                 data_root: Path, subset: int | None = None,
                 sampling: str = "dataset_balanced",
                 validity_masking: bool = False):
        self.length = length
        self.validity_masking = validity_masking
        self.arrays: dict[str, dict[str, np.ndarray]] = {}
        self.rooms: dict[str, dict[str, list[int]]] = {}
        for dataset in datasets:
            data = np.load(data_root / f"{dataset}.npz", mmap_mode="r", allow_pickle=False)
            indices = np.flatnonzero(data["split"].astype(str) == split)
            grouped: dict[str, list[int]] = defaultdict(list)
            for index in indices:
                grouped[str(data["room_id"][index])].append(int(index))
            if grouped:
                self.arrays[dataset] = {key: data[key] for key in data.files}
                self.rooms[dataset] = dict(grouped)
        if not self.rooms:
            raise RuntimeError(f"no {split} records for {datasets}")
        self.k_available = min(a["observation"].shape[1] for a in self.arrays.values())
        self.sampling = sampling
        self.buckets: dict[str, dict[int, list[int]]] = {}
        self.decay: dict[str, dict[int, dict]] = {}
        if sampling == "rt60_balanced":
            from claprir.metrics.deconvolution import decay_estimate
            for dataset, arrays in self.arrays.items():
                grouped: dict[int, list[int]] = defaultdict(list)
                self.decay[dataset] = {}
                for room, records in self.rooms[dataset].items():
                    for index in records:
                        # Full stored RIR, not arrays["rir"][index, :self.length].
                        estimate = decay_estimate(
                            np.asarray(arrays["rir"][index], np.float32))
                        value = estimate["reverberation_time_s"]
                        bucket = (len(RT60_EDGES) if estimate["censored"]
                                  else int(np.searchsorted(RT60_EDGES, value)))
                        grouped[bucket].append(index)
                        self.decay[dataset][index] = estimate
                self.buckets[dataset] = dict(grouped)
        elif sampling != "dataset_balanced":
            raise ValueError(f"unknown sampling mode {sampling}")
        # The overfit check needs a small, fixed, fully deterministic pool.
        self.fixed: list[tuple[str, int, int]] | None = None
        if subset is not None:
            pool = [(dataset, index)
                    for dataset in sorted(self.rooms)
                    for room in sorted(self.rooms[dataset])
                    for index in self.rooms[dataset][room]]
            if len(pool) < subset:
                raise RuntimeError(f"requested {subset} examples, only {len(pool)} available")
            self.fixed = [(dataset, index, 0) for dataset, index in pool[:subset]]

    def example(self, dataset: str, index: int, clap: int) -> tuple[np.ndarray, np.ndarray]:
        arrays = self.arrays[dataset]
        target = np.asarray(arrays["rir"][index, :self.length], np.float32)
        observation = np.asarray(arrays["observation"][index, clap, :self.length], np.float32)
        return target, observation

    def validity(self, dataset: str, index: int) -> np.ndarray:
        """1 where the target is real, 0 where the source recording had ended.

        Measured from the FULL stored row, not the crop, so the answer does not
        change with the horizon. A synthetic provider is valid throughout: its
        zeros are the target, not a gap.
        """
        if dataset in self.SYNTHETIC_PROVIDERS:
            return np.ones(self.length, np.float32)
        row = np.asarray(self.arrays[dataset]["rir"][index], np.float64)
        nonzero = np.nonzero(row)[0]
        last = (int(nonzero[-1]) + 1) if len(nonzero) else 0
        mask = np.zeros(self.length, np.float32)
        mask[:min(last, self.length)] = 1.0
        return mask

    def bucket_manifest(self) -> list[dict]:
        """Per-record decay metadata behind the stratification, for auditing."""
        rows = []
        for dataset, buckets in self.buckets.items():
            for bucket, records in sorted(buckets.items()):
                for index in records:
                    estimate = self.decay[dataset][index]
                    rows.append({"dataset": dataset, "record": int(index),
                                 "bucket": bucket, **estimate})
        return rows

    def sample(self, batch_size: int, rng: np.random.RandomState
               ) -> tuple[torch.Tensor, torch.Tensor]:
        targets, observations, masks = [], [], []
        datasets = sorted(self.rooms)
        for _ in range(batch_size):
            if self.fixed is not None:
                dataset, index, clap = self.fixed[rng.randint(len(self.fixed))]
            elif self.sampling == "rt60_balanced":
                dataset = datasets[rng.randint(len(datasets))]
                keys = sorted(self.buckets[dataset])
                records = self.buckets[dataset][keys[rng.randint(len(keys))]]
                index = records[rng.randint(len(records))]
                clap = rng.randint(self.k_available)
            else:
                dataset = datasets[rng.randint(len(datasets))]
                room = sorted(self.rooms[dataset])[rng.randint(len(self.rooms[dataset]))]
                records = self.rooms[dataset][room]
                index = records[rng.randint(len(records))]
                clap = rng.randint(self.k_available)
            target, observation = self.example(dataset, index, clap)
            targets.append(target[None])
            observations.append(observation[None])
            if self.validity_masking:
                masks.append(self.validity(dataset, index)[None])
        if self.validity_masking:
            return (torch.from_numpy(np.asarray(targets)),
                    torch.from_numpy(np.asarray(observations)),
                    torch.from_numpy(np.asarray(masks)))
        return (torch.from_numpy(np.asarray(targets)),
                torch.from_numpy(np.asarray(observations)))


def target_scale(config: RunConfig, data_root: Path | None = None) -> float:
    """The scalar the targets are divided by, fitted on the TRAINING split.

    ``peak`` returns 1.0 and touches nothing. ``global_rms`` returns
    ``a = mean_i RMS(h_i)`` over the training RIRs of the configured mixture,
    which is the whiteboard's dataset-level scale.

    Training split only. Fitting it on validation or test would put information
    from the scored set into the coordinate system the model is trained in.
    """
    if config.target_normalisation == "peak":
        return 1.0
    if config.target_normalisation != "global_rms":
        raise ValueError(f"unknown target_normalisation "
                         f"{config.target_normalisation!r}")
    root = Path(data_root or DEFAULT_DATA_ROOT)
    values = []
    for provider in config.training_datasets:
        path = root / f"{provider}.npz"
        if not path.exists():
            continue
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        for index in np.flatnonzero(data["split"].astype(str) == "train"):
            rir = np.asarray(data["rir"][index, :config.signal_length], np.float64)
            values.append(float(np.sqrt(np.mean(rir ** 2))))
    if not values:
        raise SystemExit(f"no training RIRs under {root} for "
                         f"{config.training_datasets}; cannot fit a scale")
    return float(np.mean(values))


def build_model(config: RunConfig, run_root: Path | None = None,
                init_root: Path | None = None) -> torch.nn.Module:
    hybrid = HybridConfig(signal_length=config.signal_length, k_max=config.k_max,
                          nf=config.nf, analysis=config.analysis,
                          input_compression=config.input_compression,
                          spectral_weight=config.spectral_weight,
                          spectral_exponent=config.spectral_exponent)
    if config.architecture == "hybrid":
        return HybridDirectRIRRegression(hybrid, stft_weight=config.stft_weight)
    if config.architecture == "hybrid_weighted":
        return HybridWeightedRegression(hybrid, stft_weight=config.stft_weight)
    if config.architecture == "hybrid_flow":
        return HybridDirectRIRFlow(hybrid, stft_weight=config.stft_weight)
    if config.architecture == "hybrid_set_late":
        model = HybridSetLate(hybrid, stft_weight=config.stft_weight)
        source = replace(config, architecture="hybrid")
        path = checkpoint_path(source, config.updates,
                               init_root or run_root or DEFAULT_RUN_ROOT)
        if not path.exists():
            raise FileNotFoundError(
                f"hybrid_set_late needs the matched regression checkpoint {path}")
        model.load_regression_backbone(
            torch.load(path, map_location="cpu", weights_only=False)["model"])
        return model
    if config.architecture == "hybrid_early_flow":
        model = HybridEarlyFlow(hybrid, stft_weight=config.stft_weight)
        # The late branch is not retrained: its weights come from the matched
        # regression run (same subset, seed, updates, batch) and are frozen.
        source = replace(config, architecture="hybrid")
        # Read-only: the frozen backbone may live on shared storage while new
        # checkpoints are written to fast local disk.
        path = checkpoint_path(source, config.updates,
                               init_root or run_root or DEFAULT_RUN_ROOT)
        if not path.exists():
            raise FileNotFoundError(
                f"hybrid_early_flow needs the matched regression checkpoint {path}; "
                "train the 'hybrid' arm for this seed first")
        model.load_regression_backbone(
            torch.load(path, map_location="cpu", weights_only=False)["model"])
        return model
    return WaveNetDirectRIRRegression(k_max=config.k_max, stft_weight=config.stft_weight)


def checkpoint_path(config: RunConfig, update: int, run_root: Path) -> Path:
    return run_root / config.run_name / f"model_updates{update}.pt"


#: Never deleted: 500 is the budget at which the previous gate was declared a
#: failure, so the matched-budget claim needs it.
KEEP_CHECKPOINTS = (500,)
#: A rolling checkpoint is overwritten on a TIME interval, not an update count.
#: Periodic external termination has been observed in this execution environment
#: on a roughly 15-minute cadence (SIGTERM at 18:15:35, 18:30:34, 18:45:34,
#: 19:00:35). The source is NOT established -- no scheduler, cgroup or platform
#: policy was inspected -- so this is recorded as an observation, not a diagnosis.
#: The engineering response is the same either way. An update-based interval
#: loses a different amount of work depending on how fast the arm runs: 200
#: updates is ~2 min at the 250 ms horizon but ~4.8 min at 1 s, a third of the
#: window. Seconds make the worst case uniform.
#: Milestones remain the record and are pruned; this one is never pruned.
ROLLING_SECONDS = 120
ROLLING_NAME = "model_rolling.pt"


def prune_checkpoints(config: RunConfig, keep_update: int, run_root: Path) -> None:
    """Drop superseded checkpoints; a 26.5 M-parameter model is ~106 MB each."""
    for path in (run_root / config.run_name).glob("model_updates*.pt"):
        update = int(path.stem.removeprefix("model_updates"))
        if update in KEEP_CHECKPOINTS or update in (keep_update, config.updates):
            continue
        path.unlink(missing_ok=True)


def milestones_for(config: RunConfig) -> tuple[int, ...]:
    """Checkpoint schedule; always includes the final update so it is loadable.

    ``500`` is kept because it is the budget at which the previous direct-RIR
    gate was declared a failure, and the matched-budget claim needs that point.
    """
    return tuple(sorted({m for m in MILESTONES if m <= config.updates} | {config.updates}))


def enable_determinism() -> None:
    """Make training bitwise reproducible from its seed.

    Off by default: every checkpoint in this programme was trained without it,
    and turning it on silently would make new runs incomparable to old ones for
    a reason nobody could see. Measured effect, in reports/training_determinism:
    two same-seed runs diverge at update 1 without it and are identical for 100
    updates with it.

    CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialises, so this warns
    rather than pretending when it is already too late.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("warning: CUBLAS_WORKSPACE_CONFIG is not set to :4096:8; cuBLAS "
              "reductions may still vary between processes", flush=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train(config: RunConfig, device: torch.device, data_root: Path,
          run_root: Path, resume: bool = True,
          init_root: Path | None = None, store=None,
          deterministic: bool = False) -> Path:
    if deterministic:
        enable_determinism()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # ``store`` is injected only by the joint multi-clap arm, whose sampler
    # yields a third element (the presence mask). Left None, this is the frozen
    # single-clap path, unchanged.
    if store is None:
        store = SingleClapStore(config.training_datasets, "train", config.signal_length,
                                data_root, config.subset, config.sampling,
                                validity_masking=config.validity_masking)
    model = build_model(config, run_root, init_root).to(device)
    # One scalar, fitted on the training split, dividing every target. 1.0 and a
    # no-op under the frozen "peak" default.
    scale = target_scale(config, data_root)
    if scale != 1.0:
        print(f"target_normalisation={config.target_normalisation}: "
              f"dividing targets by a={scale:.6f}", flush=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate)
    rng = np.random.RandomState(config.seed)
    milestones = milestones_for(config)
    output = run_root / config.run_name
    output.mkdir(parents=True, exist_ok=True)
    log, started = [], time.perf_counter()

    # Resume from the newest milestone so an interrupted multi-hour run does not
    # restart from zero.  The sampler is re-drawn from the same seeded stream and
    # fast-forwarded, so a resumed run sees the same batch sequence as an
    # uninterrupted one -- otherwise resumption would silently change the
    # experiment rather than continue it.
    start = 0
    if resume:
        done = [m for m in milestones if checkpoint_path(config, m, run_root).exists()]
        best = max(done) if done else 0
        source = checkpoint_path(config, best, run_root) if done else None
        # The rolling copy is usually newer than the newest milestone; prefer it.
        rolling = output / ROLLING_NAME
        if rolling.exists():
            try:
                update = int(torch.load(rolling, map_location="cpu",
                                        weights_only=False)["update"])
                if update > best:
                    best, source = update, rolling
            except Exception as error:      # truncated by a kill mid-write
                print(f"ignoring unreadable rolling checkpoint: {error}", flush=True)
        if source is not None:
            start = best
            payload = torch.load(source, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            rng_state = payload.get("rng")
            if rng_state is not None:
                torch.set_rng_state(rng_state["torch"].cpu()
                                    if torch.is_tensor(rng_state["torch"])
                                    else rng_state["torch"])
                if rng_state.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([t.cpu() for t in rng_state["cuda"]])
                if rng_state.get("numpy") is not None:
                    np.random.set_state(rng_state["numpy"])
                if rng_state.get("python") is not None:
                    random.setstate(rng_state["python"])
            else:
                # Checkpoints written before this was recorded. Resuming from one
                # cannot restore the trajectory, so say so instead of implying
                # the resumed run continues the original.
                print(f"{config.run_name}: checkpoint has no RNG state; the "
                      "resumed noise trajectory will differ from an "
                      "uninterrupted run", flush=True)
            # ``advance`` is the no-copy equivalent of ``sample`` where a store
            # provides one; it consumes the identical random draws, so the
            # resumed batch stream is unchanged.
            step = getattr(store, "advance", None) or store.sample
            for _ in range(start * config.accumulate):
                step(config.batch_size, rng)
            print(f"{config.run_name} resuming from update {start} "
                  f"({source.name})", flush=True)
            # Carry the previous attempts' rows forward. `log` is written with
            # write_csv, which truncates, so a resumed run used to replace the
            # whole history with whatever it logged after the restart. In an
            # environment that kills jobs every few minutes that leaves a
            # training curve covering the last few hundred updates and nothing
            # else -- measured: the compression ablation's arms kept 19 rows of
            # 20000, all after update 19550. Rows past `start` are dropped
            # rather than kept, because they belong to work the resumed
            # checkpoint does not contain.
            history = output / "training.csv"
            if history.exists():
                with history.open() as handle:
                    for row in csv.DictReader(handle):
                        if int(row["update"]) <= start:
                            log.append({key: (int(value) if key == "update"
                                              else float(value))
                                        for key, value in row.items()})
                print(f"{config.run_name} carried {len(log)} logged updates "
                      f"forward from the previous attempt", flush=True)

    model.train()
    if hasattr(model, 'backbone'):
        model.backbone.tf_stage.eval()
    last_rolling = time.perf_counter()
    for update in range(start + 1, config.updates + 1):
        optimizer.zero_grad()
        for micro in range(config.accumulate):
            batch = store.sample(config.batch_size, rng)
            target, observation = batch[0], batch[1]
            # Named, not `mask`. The store returns a TEMPORAL VALIDITY mask when
            # --validity-masking is on; the third positional argument of loss()
            # is the multi-clap PRESENCE mask. Those were the same slot once, and
            # a validity mask handed to a regression arm was silently consumed as
            # a presence mask -- no error, no warning, a run named _vm whose loss
            # was never masked at all.
            validity = batch[2].to(device) if len(batch) > 2 else None
            loss, components = model.loss(target.to(device) / scale,
                                          observation.to(device),
                                          validity_mask=validity)
            # Scale so the accumulated gradient equals the one a single batch of
            # batch_size*accumulate would have produced.
            (loss / config.accumulate).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if update == 1 or update % 25 == 0:
            log.append({"update": update, "loss": float(loss.detach()), **components,
                        "elapsed_seconds": time.perf_counter() - started})
        if update % 250 == 0 or update == 1:
            print(config.run_name, update, f"{float(loss.detach()):.5f}", flush=True)
        due = time.perf_counter() - last_rolling >= ROLLING_SECONDS
        if update in milestones or due:
            payload = {"model": model.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "config": asdict(config), "update": update,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "git_commit": git_commit(),
                       # Always recorded, unlike the commit: it does not need a
                       # repository to be present at training time.
                       "source_fingerprint": source_fingerprint(),
                       # Without this a resumed run re-seeds torch on entry and
                       # walks a DIFFERENT noise trajectory from the point of
                       # resume. Measured: a run resumed at update 50 diverges
                       # in CUDA RNG state at exactly update 51. It only bites
                       # objectives that draw from the torch RNG -- the flow
                       # arms sample noise and time every update; the
                       # regression loss samples nothing -- which is why it went
                       # unnoticed. See reports/training_determinism.
                       "rng": {"python": random.getstate(),
                               "torch": torch.get_rng_state(),
                               "cuda": (torch.cuda.get_rng_state_all()
                                        if torch.cuda.is_available() else None),
                               "numpy": np.random.get_state()}}
            # Write the rolling copy via a temporary file and rename, so a kill
            # part-way through a save cannot leave a truncated checkpoint that
            # then fails to load on resume.
            rolling = output / ROLLING_NAME
            temporary = rolling.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(rolling)
            last_rolling = time.perf_counter()
            if update in milestones:
                torch.save(payload, checkpoint_path(config, update, run_root))
                prune_checkpoints(config, update, run_root)
            write_csv(output / "training.csv", log)
    write_csv(output / "training.csv", log)
    (output / "config.resolved.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    return checkpoint_path(config, milestones[-1], run_root)


def load_model(config: RunConfig, update: int, device: torch.device,
               run_root: Path, init_root: Path | None = None) -> torch.nn.Module:
    path = checkpoint_path(config, update, run_root)
    if not path.exists():
        raise FileNotFoundError(
            f"missing checkpoint {path}; train this configuration before evaluating")
    model = build_model(config, run_root, init_root).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()
    return model


def is_stochastic(config: RunConfig) -> bool:
    return config.architecture in ("hybrid_flow", "hybrid_early_flow")


def predict_with_seed(model, observations: torch.Tensor, config: RunConfig,
                      seed: int, device: torch.device,
                      scale: float = 1.0) -> np.ndarray:
    """One prediction; flow arms draw their initial noise from a fixed seed.

    ``scale`` undoes ``target_normalisation``. A model trained on h/a predicts
    in those coordinates, so the prediction is multiplied back before it reaches
    any metric -- otherwise a normalised arm would be scored against unnormalised
    targets and would look catastrophically wrong for a reason that is purely
    bookkeeping.
    """
    if not is_stochastic(config):
        return model.predict(observations).cpu().numpy()[:, 0] * scale
    generator = torch.Generator(device=device).manual_seed(seed)
    return (model.predict(observations, generator=generator).cpu().numpy()[:, 0]
            * scale)


def selected_lambdas() -> dict[str, float]:
    """Validation-selected regularization, per excitation, from the operator audit.

    Never fall back to the ``1e-2`` default of
    ``claprir.metrics.deconvolution.regularized_deconvolution``: the audit shows it
    is three orders of magnitude above the validation optimum for true x and
    degrades that ceiling by 12x.  A baseline computed at an unselected lambda
    would understate every competitor the hybrid is measured against.
    """
    path = Path("reports/deconvolution_audit/results/selected_lambda.json")
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run claprir.analysis.regularization_selection first "
            "so the baselines use a validation-selected lambda")
    return json.loads(path.read_text())["selected"]


def crop_deconvolution(observation: np.ndarray, length: int,
                       relative_lambda: float) -> np.ndarray:
    """The meeting's weak baseline: cut away the first 3 ms and deconvolve with it."""
    clean = np.zeros(length, np.float32)
    crop = min(round(.003 * SAMPLE_RATE), len(observation))
    clean[:crop] = observation[:crop]
    return regularized_deconvolution(observation[:length], clean, length, relative_lambda)


#: The decomposition that located the residual: everything after 10 ms is fit
#: well; the failure is the direct arrival and the first reflections.
WINDOWS_MS = ((0, 10), (10, 70), (70, None))


def window_metrics(reference: np.ndarray, estimate: np.ndarray,
                   sample_rate: int = SAMPLE_RATE) -> dict[str, float]:
    """NRMSE in each diagnostic window, plus direct-path descriptors."""
    from claprir.metrics.deconvolution import direct_path_diagnostics
    out = {}
    for start_ms, stop_ms in WINDOWS_MS:
        start = round(start_ms * sample_rate / 1000)
        stop = len(reference) if stop_ms is None else round(stop_ms * sample_rate / 1000)
        window = reference[start:stop]
        error = estimate[start:stop] - window
        name = f"{start_ms}_{stop_ms if stop_ms else 'end'}ms"
        out[f"nrmse_{name}"] = float(np.sqrt(np.mean(error ** 2))
                                     / (np.sqrt(np.mean(window ** 2)) + 1e-8))
    out.update(direct_path_diagnostics(reference, estimate, sample_rate))
    return out


def early_late_metrics(reference: np.ndarray, estimate: np.ndarray,
                       edge: int) -> dict[str, float]:
    """Split the waveform error at the mixing time -- the hybrid's design claim."""
    def nrmse(a, b):
        return float(np.sqrt(np.mean((b - a) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-8))
    return {"early_nrmse": nrmse(reference[:edge], estimate[:edge]),
            "late_nrmse": nrmse(reference[edge:], estimate[edge:]),
            "early_edc_rmse_db": edc_rmse(reference[:edge], estimate[:edge])}


@torch.no_grad()
def evaluate(config: RunConfig, update: int, device: torch.device, datasets: tuple[str, ...],
             data_root: Path, run_root: Path, split: str = "test",
             max_examples: int = 16, with_baselines: bool = True,
             init_root: Path | None = None) -> list[dict]:
    # init_root matters for architectures that load a frozen stage from a
    # different tree than the one they write to (hybrid_early_flow reads its
    # frozen TF stage from the shared runs directory). Omitting it here made
    # every held-out evaluation raise, while training carried on regardless.
    model = load_model(config, update, device, run_root, init_root)
    # Undoes target_normalisation so every arm is scored in one domain.
    scale = target_scale(config, data_root)
    edge = HybridConfig().mixing_time_samples
    length = config.signal_length
    lambdas = selected_lambdas()
    # The strongest existing practical route: estimate the clap, then invert.
    # Loaded once; a missing checkpoint is a hard error, never a retrain.
    excitation_estimator = load_estimator(device) if with_baselines else None
    rows = []
    for dataset in datasets:
        data = np.load(data_root / f"{dataset}.npz", mmap_mode="r", allow_pickle=False)
        indices = np.flatnonzero(data["split"].astype(str) == split)[:max_examples]
        for index in indices:
            target = np.asarray(data["rir"][index, :length], np.float32)
            clean = np.asarray(data["clean"][index, 0], np.float32)
            observation = np.asarray(data["observation"][index, 0, :length], np.float32)
            common = {"evaluation_split": split, "dataset": dataset,
                      "room_id": str(data["room_id"][index]),
                      "sample_id": str(data["record_id"][index]),
                      "horizon_ms": config.horizon_ms, "k": 1, "update": update,
                      "seed": config.seed}
            tensor = torch.from_numpy(observation)[None, None].to(device)
            for inference_seed in (INFERENCE_SEEDS if is_stochastic(config) else (0,)):
                tick = time.perf_counter()
                estimate = predict_with_seed(model, tensor, config,
                                             inference_seed, device, scale)[0]
                elapsed = 1000 * (time.perf_counter() - tick)
                rows.append({"model": f"{config.architecture}_direct_rir_regression",
                             **common, "inference_seed": inference_seed,
                             "inference_ms": elapsed,
                             **rir_metrics(target, estimate, clean),
                             **early_late_metrics(target, estimate, edge),
                             **window_metrics(target, estimate)})
            if not with_baselines:
                continue
            estimator_model, estimator_config = excitation_estimator
            estimated_x = estimated_excitation(estimator_model, estimator_config,
                                               observation, length, device)
            padded_clean = np.zeros(length, np.float32)
            padded_clean[:min(length, len(clean))] = clean[:length]
            for name, baseline in (
                # Weak baseline the meeting named.
                ("crop_3ms_deconvolution",
                 crop_deconvolution(observation, length, lambdas["trunc_3ms"])),
                # Strong baseline: the best excitation front end on record.
                ("complex_regression_x_deconvolution",
                 regularized_deconvolution(observation, estimated_x, length,
                                           lambdas["complex_regression_x"])),
                # Ceiling of the inverse operator.
                ("true_x_deconvolution",
                 regularized_deconvolution(observation, padded_clean, length,
                                           lambdas["true_x"])),
                ("zero_predictor", np.zeros(length, np.float32)),
            ):
                rows.append({"model": name, **common, "inference_seed": 0,
                             "inference_ms": 0.,
                             **rir_metrics(target, baseline[:length], clean),
                             **early_late_metrics(target, baseline[:length], edge),
                             **window_metrics(target, baseline[:length])})
    suffix = "" if split == "test" else f"_{split}"
    write_csv(REPORT_ROOT / "results" / f"{config.run_name}_u{update}{suffix}_per_example.csv",
              rows)
    return rows


@torch.no_grad()
def evaluate_training_examples(config: RunConfig, update: int, device: torch.device,
                               data_root: Path, run_root: Path,
                               init_root: Path | None = None) -> list[dict]:
    """Exact-training-example evaluation -- the overfit gate."""
    if config.subset is None:
        raise ValueError("the overfit gate requires a fixed training subset")
    model = load_model(config, update, device, run_root, init_root)
    scale = target_scale(config, data_root)
    store = SingleClapStore(config.training_datasets, "train", config.signal_length,
                            data_root, config.subset)
    edge = HybridConfig().mixing_time_samples
    rows = []
    seeds = INFERENCE_SEEDS if is_stochastic(config) else (0,)
    for dataset, index, clap in store.fixed:
        target, observation = store.example(dataset, index, clap)
        clean = np.asarray(store.arrays[dataset]["clean"][index, clap], np.float32)
        tensor = torch.from_numpy(observation)[None, None].to(device)
        # One row per inference seed for stochastic arms, so the spread is visible
        # rather than a single draw standing in for the method.
        for inference_seed in seeds:
            estimate = predict_with_seed(model, tensor, config, inference_seed, device, scale)[0]
            rows.append({"model": f"{config.architecture}_direct_rir_regression",
                         "inference_seed": inference_seed,
                         "evaluation_split": "exact_train", "dataset": dataset,
                         "room_id": str(store.arrays[dataset]["room_id"][index]),
                         "sample_id": str(store.arrays[dataset]["record_id"][index]),
                         "update": update, "seed": config.seed,
                         **rir_metrics(target, estimate, clean),
                         **early_late_metrics(target, estimate, edge),
                         **window_metrics(target, estimate)})
    write_csv(REPORT_ROOT / "results" / f"{config.run_name}_u{update}_exact_train.csv", rows)
    return rows


def summarize(rows: list[dict], keys=("nrmse", "early_nrmse", "late_nrmse", "env_corr",
                                      "lsd_db", "edc_rmse_db", "roundtrip_nrmse")
              ) -> list[dict]:
    out = []
    for model in sorted({r["model"] for r in rows}):
        for dataset in sorted({r["dataset"] for r in rows if r["model"] == model}):
            subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
            row = {"model": model, "dataset": dataset, "n": len(subset)}
            for key in keys:
                values = [r[key] for r in subset if key in r and np.isfinite(r[key])]
                if values:
                    row[f"{key}_median"] = float(np.median(values))
                    row[f"{key}_mean"] = float(np.mean(values))
            out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "overfit", "evaluate"), required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--training-datasets", nargs="+", default=["shoebox"])
    parser.add_argument("--test-datasets", nargs="+",
                        default=["shoebox", "mit", "but", "ace", "openair"])
    parser.add_argument("--horizon-ms", type=int, default=250)
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset", type=int)
    parser.add_argument("--seed", type=int, default=42_001)
    parser.add_argument("--nf", type=int, default=128)
    parser.add_argument("--target-normalisation", dest="target_normalisation",
                        choices=("peak", "global_rms"), default="peak",
                        help="peak keeps the shards as they are; global_rms "
                             "divides targets by one training-fitted scalar")
    parser.add_argument("--spectral-weight", dest="spectral_weight", type=float,
                        default=0.0,
                        help="weight on the multi-resolution STFT term; 0 is the "
                             "frozen default, 0.02 is the registered Step 4A value")
    parser.add_argument("--stft-weight", dest="stft_weight", type=float,
                        default=.25,
                        help="weight on the single-resolution compressed-STFT "
                             "term; 0.25 is the frozen default and 0.0 is the "
                             "pure flow-matching objective (Eloi arm)")
    parser.add_argument("--validity-masking", dest="validity_masking",
                        action="store_true",
                        help="supervise only where the target is real; past a "
                             "record's last recorded sample the shard zeros are "
                             "absent data, not silence")
    parser.add_argument("--spectral-exponent", dest="spectral_exponent",
                        type=float, default=COMPRESSION_EXPONENT,
                        help="exponent inside the auxiliary spectral term; 2/3 "
                             "is the frozen default and 1.0 makes it the plain "
                             "STFT-magnitude MSE. Independent of "
                             "--input-compression, which warps the model input")
    parser.add_argument("--input-compression", type=float,
                        default=COMPRESSION_EXPONENT,
                        help="magnitude exponent for the 2-D stage input; "
                             "2/3 is the frozen default, 1.0 removes the "
                             "compression (representation ablation)")
    parser.add_argument("--sampling", default="dataset_balanced",
                        choices=("dataset_balanced", "rt60_balanced"))
    parser.add_argument("--analysis", default="stft", choices=("stft", "multires"))
    parser.add_argument("--accumulate", type=int, default=1,
                        help="micro-batches per optimiser step; effective batch "
                             "is --batch-size times this")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--evaluate-update", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="bitwise-reproducible training; set "
                             "CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching")
    parser.add_argument("--init-run-root", type=Path,
                        help="where to READ a frozen backbone from; defaults "
                             "to --run-root. Lets checkpoints be written to "
                             "local disk while initialising from shared storage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    config = RunConfig(architecture=args.architecture,
                       training_datasets=tuple(args.training_datasets),
                       seed=args.seed, horizon_ms=args.horizon_ms,
                       updates=args.updates, batch_size=args.batch_size,
                       subset=args.subset, nf=args.nf, sampling=args.sampling,
                       analysis=args.analysis, accumulate=args.accumulate,
                       # Both of these were parsed but never reached the config,
                       # so every arm silently trained at the defaults. A flag
                       # that argparse accepts and the model never sees is worse
                       # than no flag: it produces a table of arm names whose
                       # rows are all the same experiment.
                       input_compression=args.input_compression,
                       spectral_weight=args.spectral_weight,
                       stft_weight=args.stft_weight,
                       spectral_exponent=args.spectral_exponent,
                       validity_masking=args.validity_masking,
                       target_normalisation=args.target_normalisation)
    update = args.evaluate_update or milestones_for(config)[-1]
    if args.stage in {"train", "overfit"}:
        if args.stage == "overfit" and config.subset is None:
            raise ValueError("--subset is required for the overfit stage")
        train(config, device, args.data_root, args.run_root,
              resume=not args.no_resume, init_root=args.init_run_root,
              deterministic=args.deterministic)
    if args.stage == "overfit":
        rows = evaluate_training_examples(config, update, device, args.data_root,
                                          args.run_root, args.init_run_root)
        summary = summarize(rows)
        print(json.dumps(summary, indent=2))
    if args.stage == "evaluate":
        rows = evaluate(config, update, device, tuple(args.test_datasets),
                        args.data_root, args.run_root, args.evaluation_split,
                        args.max_examples, init_root=args.init_run_root)
        print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
