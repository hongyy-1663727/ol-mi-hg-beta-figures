# OL_MI HG/β figures — B6

Figure-generation code for the high-gamma (HG, 70–150 Hz) and beta (β, 13–30 Hz)
analysis of subject **B6**'s open-loop motor-imagery (OL_MI) session (20260708,
256-channel ECoG @ 1 kHz).

Three 10 s open-loop motor-imagery tasks:

| Task | Structure | Pacing |
|---|---|---|
| **ColorPaced** | one sustained imagined action | external 0.5 Hz color cue (green→grey at 2/4/6/8 s) |
| **SelfPaced** | one sustained imagined action | internal, no external cue |
| **RandSwitch** | 7 actions, switched every 2 s | switch-driven |

## Figures

`ol_mi_figures.py` produces seven report figures from five functions (the three
per-task "own-ROI" panels are one function called per task):

| Function | Figure | Input |
|---|---|---|
| `fig_single_channel()` | single-trial, 5-channel HG/β across the 3 tasks | `data/ol_mi_ch5.npz` |
| `fig_channel_grid()` | HG execute−rest selection map on the ChMapB2 array | `data/ol_mi_env.npz` |
| `fig_own_roi(task)` | per-task 3×3 channel-level panel, each band on its own ROI | `data/ol_mi_bothroi.npz` |
| `fig_roi_grid()` | HG-ROI vs β-ROI electrode layout on ChMapB2 | `data/ol_mi_env.npz` |
| `fig_pace_detection()` | per-trial individual-pace / arrhythmia test | `data/ol_mi_env.npz` |

`ol_mi_colorpaced_per_action.py` produces a standalone eighth figure — the
ColorPaced response broken out across all 7 imagined actions.

All input checkpoints are produced from the raw `.mat` recordings by `extract.py`
(see *Pipeline* below). They are **not** included in this repo — run `extract.py`
on the acquisition server to generate `data/`, then the figure scripts run anywhere.

## Signal convention

Every figure uses the same envelope pipeline, applied in this order (see
`extract.py`):

> common-median reference across channels → 1 s reflect-pad (to suppress
> filter/Hilbert edge transients, cropped after) → 60/120/180 Hz notch →
> 70–150 Hz (HG) / 13–30 Hz (β) 4th-order Butterworth band-pass → Hilbert
> analytic-amplitude envelope → per-channel z-score to that trial's rest baseline →
> resampled to a 100 Hz grid → 200 ms moving-average smoothing for display.

ROIs are the 40 channels with the largest execute−rest HG *increase* (HG ROI) and
β *decrease* (β ROI), defined once over the sustained-task trials and reused
everywhere.

## Data

No data is committed to this repo — subject recordings are not redistributed here.
`extract.py` regenerates everything under `data/` from the raw `.mat` on the
acquisition server:

| File | What it is |
|---|---|
| `ol_mi_env.npz` | per-trial ROI-mean envelopes + per-channel execute−rest maps (the frozen checkpoint) |
| `ol_mi_labels.json` | metadata: sample rate, bands, ROI channel lists, per-file trial records |
| `ol_mi_ch5.npz` / `.json` | one exemplar trial per task, 5 channels at 1 kHz (single-channel figure) |
| `ol_mi_bothroi.npz` | per-task channel-average + per-trial ROI-mean envelopes, both bands (own-ROI figures) |

The one figure `extract.py` does not build is the ColorPaced per-action figure; its
reduced input (`ol_mi_cp_pa_fig.npz`) is produced by the extract/reduce stages inside
`ol_mi_colorpaced_per_action.py`, which read the same raw `.mat`.

The raw recordings themselves (~700 MB/run) live on the acquisition server.

## Pipeline

Two stages: **raw → checkpoints** (server, needs the `.mat`) and
**checkpoints → figures** (anywhere).

```
raw .mat ──extract.py──▶ data/*.npz + *.json ──ol_mi_figures.py──▶ figures
```

`extract.py` is the full raw-reading stage. It applies the locked envelope pipeline
(below), defines the ROI, and writes the three checkpoints. The standalone
`ol_mi_colorpaced_per_action.py` additionally carries its own raw → reduce → plot
path inline for the per-action figure.

## Run

```bash
pip install -r requirements.txt

# Stage 1 (on the server, needs the raw .mat) — generate the checkpoints into data/:
python extract.py --root /mnt/DataDrive/B6Data/20260708 --out data

# Stage 2 (runs anywhere data/ exists) — render the figures:
python ol_mi_figures.py                 # -> 7 figures
python ol_mi_colorpaced_per_action.py   # -> the per-action figure
```

Figures are written as `.png` into the working directory. Reference renders are in
`examples/` (the only data-derived files committed here).

## Notes

- One subject, one session — read all results as within-session findings pending
  replication.
- Tested with numpy 1.21.5 / scipy 1.8.0 / matplotlib 3.5.1. One scipy-version
  gotcha is noted in `requirements.txt`.
