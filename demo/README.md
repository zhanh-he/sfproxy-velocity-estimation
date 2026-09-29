# Paper, figures, and listening demo

This directory holds presentation material and the evidence behind the displayed numbers.

| Path | Contents |
| --- | --- |
| `docs/` | Static listening site; `assets/` holds the 20 s MAESTRO audio/MIDI example and slide-ready SVG/JSON files |
| `analysis/` | Camera-ready Tables 1–2 as CSV, an inventory of 5090 evaluation summaries, the figure generator, analysis package, statistics, tests, and notebooks |
| `paper/` | Supplied camera-ready PDF and LaTeX source snapshot |
| `scripts/` | Scripts to export the audio example and index saved evaluations |

The site compares human audio with **Flat 64, VeloEst, Diff-Synth, and VeloEst+Diff-SFProxy** on one matched piano recording. Visitors can select a note in any colorized MIDI roll to see the same note's velocity in every approach and seek the audio players to its onset. The VeloEst and Diff-SFProxy audio/MIDI are regenerated from the two released weights. MAE badges are full-piece values for this one example. The [721-note inference CSV and figure](analysis/inference_example/README.md) document the new checkpoint comparison. Guitar correlations remain dataset-level paper results, with no final 5 s guitar audio example in this repository. See [`analysis/README.md`](analysis/README.md) and [`analysis/PROVENANCE.md`](analysis/PROVENANCE.md).

From the repository root:

```bash
python3 demo/analysis/build_figures.py
python3 -m http.server 8000 --directory demo/docs
```

Open `http://localhost:8000/`. The paper-result SVGs under `docs/assets/` are 1280 × 720; the checkpoint comparison is 1600 × 860. All can be imported into slides. The website source lives **inside `demo/docs/`** and is deployed by [`../.github/workflows/deploy-demo.yml`](../.github/workflows/deploy-demo.yml) from `camera-ready-release` after GitHub Pages is set to Actions as its publishing source.
