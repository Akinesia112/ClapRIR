# ClapRIR demo

Project page: https://akinesia112.github.io/ClapRIR/

Seven phone-recording rooms, eight modes, five repetitions per mode, with paired recorded-clap and inferred-RIR players, selected-example spectra/spectrograms, and documented room photographs. The default examples use the frozen input-only selection; the paper-linked examples reuse the exact saved paper predictions. All audio is independently peak-normalized for playback only. No reference RIR is available for the phone recordings.

## Contents

- index.html: project text and the existing academic-template layout.
- static/js/phone-demo.js: room/mode/repetition selection; no audio autoplay.
- static/assets/: verified asset bundle and provenance manifests.
- static/audio_examples.zip: six paper-selected pairs for listening.

The original template CSS and existing JavaScript are unchanged. No model inference is run by this site. Serve locally with python -m http.server and open index.html. The source/processing notes and inventory are in static/assets/README.md and static/assets/manifest.csv.

Template credit: https://github.com/eliahuhorwitz/Academic-project-page-template (derived from Nerfies). The template license does not automatically license research recordings and photographs.
