# Clap taxonomy — final specification for the two-layer clap figure

**Status: final spec.** Both layers can now be built. The corpus is resolved
(280 protocol claps across 7 rooms, all eight modes assigned) and the frozen
selection rule has produced 56 selected claps, one per room × mode, cut from the
authoritative `.m4a`. This file states what the figure is and what data feeds
it. It supersedes every earlier draft of this document.

Nothing here asserts an acoustic property of any clap mode. The mode labels
describe **hand geometry**, which is what the clap-mode reference documents.

---

## 1. The eight modes

Eight labelled clap modes, performed in this fixed order, five repeats each:

```
P1  P2  P3  |  A1  A2  A3  |  A1-  A1+
```

| group | modes | what the reference documents |
|---|---|---|
| Parallel | P1, P2, P3 | palms parallel; the three settings differ in hand offset |
| Angled | A1, A2, A3 | palms meet at an angle; the three settings differ in contact angle |
| A1 variants | A1-, A1+ | variants of the A1 geometry, one step either side of nominal A1 |

**Label the panels with the mode names and the reference's own hand-position
images. Do not construct geometric axes on top of them.** Earlier drafts of this
file drew two "controlled variables" — a monotonic offset axis for P1→P3 and a
directional angle sweep for A1→A3 — and guessed at the `-`/`+` semantics with
hedges like "most plausibly clap strength". Those were the document's
inventions, not the reference's content. Unless the clap-mode reference states a
direction, a magnitude or a varied parameter, the figure shows the eight
positions and names them; it does not claim an ordering.

If the reference *does* state them, write them from the reference. Do not ask
the recordist: an author was present at the recordings and the clap-mode
reference used to run the experiment defines these modes. This is not an open
question.

---

## 2. The figure

**Top row — hand position.** Eight panels, one photograph per mode, from the
existing clap-mode reference. Fixed camera position and angle so the panels are
comparable. Order `P1 P2 P3 | A1 A2 A3 | A1- A1+`, with the three groups
visually separated. The caption must name which reference the images are
reproduced from.

The 13 JPEGs in `data/DifferentClaps_PhoneRecording_V1/pictures/` are **room**
photographs, not hand positions. The top row is not sourced from this delivery.

**Bottom row — acoustics.** One panel per mode, column-aligned under its hand
photograph. Either the time-domain waveform or the normalised magnitude spectrum
— one choice, applied to all eight panels, with identical axis limits and
identical normalisation, or it is not a comparison.

Drawn from the **frozen selected claps** (§3). Because the same eight modes were
performed in all seven rooms, the panel must pin down its room. Two defensible
constructions:

* **single room** — all eight panels from one named room, named in the caption;
* **per-room grid** — 8 modes × 7 rooms, room axis labelled.

A panel that silently mixes rooms is not interpretable and must not be produced.

---

## 3. Which data feeds which claim

This is the part that must not be improvised. The two sets are not
interchangeable.

| purpose | data | n |
|---|---|---|
| dataset characterization, statistics, any distribution or variance claim | **all protocol claps** | **280** = 7 rooms × 8 modes × 5 repeats |
| figure panels, project-page audio, one-per-cell visual demonstration | **frozen selected claps** | **56** = 7 rooms × 8 modes |

The 56 are a one-per-cell illustration chosen by a predefined QC rule (highest
pre-clap SNR within each room × mode, clipped events excluded). **They are not a
sample to compute statistics on and must never stand in for the 280.**

Both sets are cut from the authoritative `.m4a`. Row-level index:
`reports/phone_clap_demo/results/metadata.csv` —
`source == "m4a"`, and:

* the 280: `event_type == "clap"` and `protocol_status == "normal"`;
* the 56: additionally `selected_for_demo == 1`;
* segment audio: the path in `segment_path`, under
  `reports/phone_clap_demo/results/segments/m4a/`.

The 11 non-protocol detections (room 1's 7 `extra_clap`, rooms 6/7's 3 handling
events, room 2's 1 environmental event) are retained for provenance and QC.
**They must not enter clap-mode statistics and must not appear in a mode panel.**

---

## 4. Caption constraints

* This dataset has **no paired ground-truth RIR**. The figure may show what
  different hand geometries look like and what they sound like. It may not
  claim, suggest, or be captioned in a way that implies, that any model
  recovers, matches or estimates a room response from these claps to any degree
  of accuracy.
* Do not put an RT60, a T20, a C50 or any other acoustic figure of merit on this
  plate. These recordings reach a measured noise plateau after 20–30 dB of decay
  (`reports/phone_clap_demo/README.md` §A5), so single-clap reverberation times
  are not supportable from this corpus.
* Room names and file provenance come from `rooms/room_map.csv`, which resolves
  the numbered `recordings/` names against the plain-language `raw/` names.
* Anyone counting or segmenting claps uses the `.m4a`. The `recordings/*.wav`
  are diagnostic derivatives and do not hold the same event set — see
  `rooms/README.md` note 2.

---

## 5. Open items

Neither blocks the figure.

1. **Mode semantics.** Whether the reference states a direction/magnitude for
   P1→P3 and A1→A3, and what the `-`/`+` suffix varies. If it does, label from
   it; if it does not, label positions only (§1).
2. **Room 1, two mode-level ambiguities** — which of its ev40/ev41 is A1+ repeat
   5, and what its extra ninth block of five claps was
   (`reports/phone_clap_demo/README.md` §8.5). The first affects one `repeat_id`
   inside A1+ and no count; the second concerns events excluded from the 280
   altogether. A mode panel drawn from the 56 selected claps is unaffected by
   both.
