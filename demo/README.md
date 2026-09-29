# Paper, figures, and listening demo

This directory holds presentation material and the evidence behind the displayed numbers.

| Path | Contents |
| --- | --- |
| `docs/` | Static listening site; `assets/` holds the 20 s MAESTRO audio/MIDI example and generated SVG/JSON files |
| `analysis/` | Camera-ready Tables 1–2 as CSV, an inventory of 5090 evaluation summaries, the figure generator, analysis package, statistics, tests, and notebooks |
| `paper/` | Supplied camera-ready PDF and LaTeX source snapshot |
| `scripts/` | Scripts to export the audio example and index saved evaluations |

The site compares human audio with Flat 64, Diff-Synth, and Diff-SFProxy on **one matched piano recording**. Its MAE badges are full-piece values for that example. Guitar correlation values are transcribed dataset-level results from the paper and have no corresponding final 5 s audio example in this repository. See [`analysis/README.md`](analysis/README.md) and [`analysis/PROVENANCE.md`](analysis/PROVENANCE.md).

From the repository root:

```bash
python3 demo/analysis/build_figures.py
python3 -m http.server 8000 --directory demo/docs
```

Open `http://localhost:8000/`. The SVGs under `docs/assets/` are 1280 × 720 and can be imported into slides. The website source lives **inside `demo/docs/`**. GitHub Pages needs a custom Actions deployment for a nested source directory; see [`../.github/workflows/deploy-demo.yml`](../.github/workflows/deploy-demo.yml) before changing the current Pages setting.
