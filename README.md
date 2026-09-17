# Estimation of Room Impulse Responses from Handclaps

Shih-Yu Lai<sup>1,2,3</sup> &nbsp; Kyung Yun Lee<sup>2</sup> &nbsp; Nils Meyer-Kahlen<sup>2</sup> &nbsp; Eloi Moliner<sup>2</sup> &nbsp; Bing-Yu Chen<sup>1</sup> &nbsp; Vesa Välimäki<sup>2</sup>

<sup>1</sup> National Taiwan University, Taipei, Taiwan
<sup>2</sup> Acoustics Lab, DICE, Aalto University, Espoo, Finland
<sup>3</sup> MoonShine Animation Studio, Taipei, Taiwan

A handclap is an equipment-free excitation for room acoustics, but its source
waveform is unknown and varies between claps, which makes room impulse response
(RIR) estimation hard. This repository holds the code and the project website for that
work.

## Examples and listening demo

- [Project website](https://akinesia112.github.io/ClapRIR/)
- [Download six paper-selected recorded-clap / inferred-RIR pairs](https://raw.githubusercontent.com/Akinesia112/ClapRIR/main/docs/static/audio_examples.zip)
- [Complete website and asset bundle (ZIP)](https://github.com/Akinesia112/ClapRIR/archive/refs/heads/main.zip)
- [Asset inventory and provenance](docs/static/assets/README.md)
- [Per-asset manifest](docs/static/assets/manifest.csv)

The demo contains seven phone-recording rooms, eight handclap modes and five
repetitions per mode (280 paired recordings and inferred RIRs), spectra,
spectrograms, 13 documented room photographs and five anechoic examples.
Listening WAVs are mono, 44.1 kHz, peak-scaled to −1 dBFS for playback only. The
phone recordings have no paired reference RIR, so those examples illustrate
behaviour rather than measure accuracy.

## Repository layout

```
src/claprir/        the library
  models/           RIR estimators and their building blocks
  metrics/          evaluation metrics
  datasets/         corpus and shard construction
  rir/              RIR file handling and signal generation
  training/         the training entry point
  analysis/         regularization selection, clap spectrum analysis
  plotting/         shared figure style
experiments/        one script per experiment; each writes a report directory
figures/            scripts that render the manuscript figures
tables/             scripts that render the manuscript tables
docs/               the project website (GitHub Pages)
```

Names say what things do. The research tree used internal codenames (`E1`, `E3`,
`E5`, `E6`, `post_meeting_table1`, `fig1`); those are gone. The only short labels
kept are the handclap-mode labels from the paper itself (`A1`, `P1`, …), which
are part of the dataset vocabulary. `PROVENANCE.md` maps every file here back to
its origin.

## Install

```bash
pip install -e .                  # add [simulation] for the shoebox generator
```

`pyroomacoustics` is optional and only needed to regenerate simulated RIRs.

## Reproducing the paper

Datasets, checkpoints and run artifacts are **not** in git (see `.gitignore`);
scripts expect them under `data/`, `runs/` and `reports/` at the repository root.

Build the data, then run the experiments, then render figures and tables:

```bash
python -m claprir.datasets.controlled_shards      # controlled shoebox/measured shards
python -m claprir.datasets.rir_shards             # one-second multi-room shards
python -m claprir.training.train_rir_estimator --stage train \
    --architecture hybrid --training-datasets shoebox mit but ace \
    --horizon-ms 1000 --updates 20000 --batch-size 2 --accumulate 8 \
    --validity-masking --input-compression 1.0 --seed 42001

python experiments/single_clap_benchmark.py       # the main results table
python experiments/cropped_excitation_baseline.py # cropped-excitation baselines
python experiments/energy_decay_metric_recompute.py
python figures/benchmark_comparison_figures.py
python tables/build_tables.py
```

Every experiment script writes a self-describing report directory containing a
`result.json`, the per-example and per-room CSVs behind each number, and a README
stating what was run.

Three modules under `claprir.datasets` read the handclap split at import time, so
they require `data/` to be present and the repository root as the working
directory. That behaviour is inherited from the research code and was left as is.

## Not included

- **Datasets, checkpoints and report artifacts.** Too large for git; the website
  bundle above carries the audio examples.
- **The blind joint deconvolution experiment.** It is a separate line of work and
  is not part of this paper. One consequence: the `score_all` path of
  `experiments/arpege_external_evaluation.py` imports an echo-density helper from
  that experiment and will not run without it. The ARPEGE figures do not need it —
  they read precomputed results.
- **Exploratory arms** of the research tree that no manuscript figure or table
  depends on.
- **The manuscript sources.** The figure and table scripts here regenerate the
  assets the paper uses; the LaTeX itself is kept outside this repository.

## Relationship to the research repository

Code here was copied verbatim from the internal research tree. The only edits
were mechanical: module paths in import statements, the repository-root depth in
`Path(__file__).parents[N]`, and replacing one machine's absolute paths and GPU
UUID with the `CLAPRIR_RUNTIME` and `CLAPRIR_GPU_UUID` environment variables.
Numerical behaviour was checked to be unchanged: the metrics reproduce the
research tree bit for bit.

## Source and reuse terms

The repository LICENSE applies to the code. The academic website template license
and credits are preserved under `docs/`. That license does not grant unspecified
reuse rights to recordings or room photographs; see the
[asset release notes](docs/static/assets/README.md).
