# Paper, figures, and listening demo

This directory holds presentation material and the evidence behind the displayed numbers.

| Path | Contents |
| --- | --- |
| `docs/` | Static listening site; `assets/cases.json` selects 20 s examples from MAESTRO, SMD, FL, and GAPS |
| `analysis/` | Camera-ready Tables 1–2 as CSV, recovered result evidence, the figure generator, analysis package, statistics, tests, and notebooks |
| `paper/` | Supplied camera-ready PDF and LaTeX source snapshot |
| `scripts/` | [Documented tools](scripts/README.md) to extract, build, validate, and level the listening cases and index saved evaluations |

The site uses a dataset selector and 20 s case menu. Every case has the same playable positions: **00 original recording, 01 Flat 64, 02 Diff-Synth, 03 Diff-SFProxy**. There are six windows each from GAPS, FL, MAESTRO, and SMD: 24 cases and 96 aligned audio clips in total. The guitar methods use adapted 5 s checkpoints recovered from lab5090's second HDD; see the [guitar recovery record](analysis/GUITAR_RESULT_RECOVERY.md) for the archived runs, hashes, and rerun checks. The project team confirmed that its GAPS audio access covers these research demo excerpts. The [case inventory and source details](analysis/LISTENING_CASES.md) distinguish individual examples from paper aggregate scores. The [721-note VeloEst checkpoint comparison](analysis/inference_example/README.md) remains available as a separate downloadable figure and CSV.

From the repository root:

```bash
python3 demo/analysis/build_figures.py
python3 -m http.server 8000 --directory demo/docs
```

Open `http://localhost:8000/`. The paper-result SVGs under `docs/assets/` are 1280 × 720; the checkpoint comparison is 1600 × 860. All can be imported into slides. The website source lives **inside `demo/docs/`** and is deployed by [`../.github/workflows/deploy-demo.yml`](../.github/workflows/deploy-demo.yml) from `backup` after GitHub Pages is set to Actions as its publishing source.

## Figures for the repository front page

| File | Use |
| --- | --- |
| [`paper/figure0.jpg`](paper/figure0.jpg) | Compact visual introduction: piano-trained velocity estimation transferred to guitar. |
| [`paper/figure1.jpg`](paper/figure1.jpg) | Paper method overview: Diff-Synth versus Diff-SFProxy adaptation. |

Both are embedded with relative paths in the root README, so the preview follows the branch being viewed. The listening site uses the selected warm-white Gallery layout; its audio, MIDI, scores, and case selection are unchanged.
