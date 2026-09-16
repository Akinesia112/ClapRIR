# Estimation of Room Impulse Responses from Handclaps

<!-- Website release -->

## Examples and listening demo

- [Project website](https://akinesia112.github.io/ClapRIR/)
- [Download six paper-selected recorded-clap / inferred-RIR pairs](https://raw.githubusercontent.com/Akinesia112/ClapRIR/main/docs/static/audio_examples.zip)
- [Complete website and asset bundle (ZIP)](https://github.com/Akinesia112/ClapRIR/archive/refs/heads/main.zip)
- [Asset inventory and provenance](docs/static/assets/README.md)
- [Per-asset manifest](docs/static/assets/manifest.csv)
- [Paper figure: spectra and RMS envelopes](docs/static/assets/figures/waveforms/paper_spectra_and_waveforms_15khz.pdf)
- [Raw linear waveform alternative](docs/static/assets/figures/waveforms/paper_spectra_and_waveforms_linear_15khz.pdf)

The demo contains seven phone-recording rooms, eight handclap modes and five repetitions per mode (280 paired recordings and inferred RIRs), spectra, spectrograms, 13 documented room photographs and five anechoic examples. The original website template design is preserved.

Paper-linked examples reuse the frozen input-selected samples and exact saved estimates. Listening WAVs are mono, 44.1 kHz, and independently peak-scaled to -1 dBFS for playback only. No inference was rerun. The phone recordings have no paired reference RIR; these examples do not measure reconstruction accuracy.

## Local preview

Run python -m http.server --directory docs 8000, then open http://localhost:8000/. GitHub Pages uses the main branch, /docs directory.

## Source and reuse terms

The original repository LICENSE is preserved. The academic website template license and credits are preserved under docs. That software/template license does not grant unspecified reuse rights to recordings or room photographs; see [asset release notes](docs/static/assets/README.md).
