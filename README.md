# ol-mi-hg-beta-figures

Figures for the **open-loop motor-imagery high-gamma / beta-band** analysis in the
Ganguly-lab ECoG brain–computer interface (BCI) research program.

> **Acronyms** (spelled out on first use, per lab convention):
> - **OL** — Open Loop (task condition run without closed-loop decoder feedback)
> - **MI** — Motor Imagery (imagined movement, no overt motor output)
> - **HG** — High Gamma, the ~70–170 Hz broadband high-frequency neural feature
> - **Beta** — the ~13–30 Hz beta rhythm
> - **ECoG** — Electrocorticography (subdural electrode recording)

## Purpose

This repository holds the **figure-generation code and rendered figures** for the
OL-MI HG/Beta analysis. It is a standalone, sharable figures repo — kept separate from
the raw-data and pipeline repositories so the figures can be versioned and reviewed on
their own.

## Layout

```
ol-mi-hg-beta-figures/
├── README.md
├── .gitignore
├── figures/     # rendered outputs (.png / .pdf / .svg) — tracked
├── scripts/     # figure-generation scripts / notebooks
└── data/        # intermediate/derived inputs (large raw blobs are git-ignored)
```

## Data policy

Raw neural data lives on the data drives, **not** in this repo:

- Hot storage: `/mnt/DataDrive`
- Cold backup: `/media/user/New Volume`

Large binary blobs (`.mat`, `.h5`, `.npz`, `.npy`, …) are git-ignored by default —
commit **derived** figure inputs and the rendered figures, not multi-GB source recordings.

## Related repositories (siblings in `/home/user/Desktop/HongyiGithub/`)

- **BetaPaper** — manuscript, supplementary material, and figures for the BCI beta paper
- **BCISignalAnalysis_Python** — signal-analysis tooling
- **bci_plot** — plotting utilities
- **ProjectManager** — cross-machine coordination and the Obsidian knowledge-base vault
