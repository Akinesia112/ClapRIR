# ClapRIR website assets

Three groups: paper-linked examples, website-only extensions, and dataset recording material. The bundle reuses existing inputs and model estimates; no inference or training was run.

## Paper-linked material

The two main rooms use exactly the frozen paper membership: one unclipped recording for each of eight modes, chosen by highest pre-handclap SNR before inference. The meeting-room predictions come from the authoritative saved figure arrays, not substituted full-corpus cache rows. Raw arrays and per-asset hashes are included. The six-pair Slack bundle is at ../audio_examples.zip.

## Website extensions

All 280 normal phone protocol events (7 rooms x 8 modes x 5 repetitions) have paired recorded/inferred listening WAVs. The 56 preselected demo events feed the room spectra and spectrograms. All 13 existing room photographs have documented room identities and are copied unchanged. Room labels and exported filenames use Highly reverberant hallway; original source paths preserve original filenames for traceability.

## Processing

Audio: mono 44.1 kHz IEEE float WAV. Phone files preserve the complete one-second model interval and original alignment, including existing input padding where valid recording support is shorter. Each listening file alone is peak-scaled to -1 dBFS. No extra filtering, denoising, onset shifting or inference is applied. Every export is read back and checked for exact scaled sample equality, rate, duration, finite samples and no clipping. Original peaks/gains are in manifest.csv.

Website spectra: 0.1--10 kHz default, common 0--0.9-s interval, 1/12-octave mean linear FFT power, per-signal mean FFT-bin power normalization over 0.1--10 kHz. Original 15-kHz paper figures and a separately labeled existing 20-kHz diagnostic are additional assets. Spectrograms: Hann 1024, hop 128, FFT 1024, boundary zeros, no padded final STFT frame; 0--0.9 s and 0.1--10 kHz; normalization by each panel mean power; shared -20 to +30 dB. Waveforms: 0--250 ms. The supplementary P1/P2/P3 strip uses linear peak-normalized amplitude. The updated combined paper figures group each room spectra above its waveforms, with recorded-clap / inferred-RIR columns and panels (a)--(h). Both combined versions use peak-normalized 1/12-octave spectral power within 0.1--15 kHz, displayed from -40 to 0 dB. The main time-domain version uses a centered 221-sample RMS envelope with 44-sample spacing (about 5 ms / 1 ms), normalized independently to its envelope peak and displayed as RMS level (dB), -40..0 dB. The linear alternative preserves original signed digital samples without scaling, with shared -1..1 axes. The RMS representation changes only visualization; the underlying samples and predictions remain unchanged.

## Dataset material and release status

Five anechoic listening examples use the same source events as fig_clap_dataset, from participants 01, 02, 04, 05 and 07, preserving their complete retained source clip rather than the figure 20-ms crop. Individual-event clap-mode labels are unknown and are left blank. The local legacy event inventory and the newer 2540-event release description in manuscript edits are distinct; no newly released dataset statistics or mode assignments are inferred from the legacy files.

The phone audio, verified room photographs, generated figures and unlabelled anechoic examples are prepared for the user-requested public demo. The repository software license is recorded separately. No unverified pose images, invented setup photograph, unresolved paper/preprint link, or blanket dataset license is published. No paired phone reference RIR exists, and no reconstruction-accuracy claim is made. Perceptual plausibility remains a human listening judgment.

## Missing or not yet suitable for publication

- Anechoic recording setup image clap_measurement.jpeg is referenced in the manuscript but absent from the local publication/data material.
- Verified P1/P2/P3/A1/A2/A3/A1-/A1+ hand-position image files were not found locally; no substitute images or event-level pose mapping were created.
- Public full anechoic-dataset download URL not present in the inspected sources.
- Public paper/preprint URL and downloadable model checkpoint link not present in the inspected sources.
- Separate reuse/license terms for dataset audio and room photographs were not present; the software MIT license is not extended to them.

Source identifiers, checkpoint SHA-256, original peaks, gains, selected-for-paper flags and exact export paths are recorded in manifest.csv, phone_demo/all_rooms/recording_inventory.csv, and the JSON inventories. The manuscript was not modified. The combined paper figure was separately revised at the user request to add recorded-clap and inferred-RIR magnitude-in-dB rows; its source signals and predictions are unchanged.
