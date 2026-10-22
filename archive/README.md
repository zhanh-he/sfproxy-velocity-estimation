# Backup-only archive

This folder keeps presentation candidates and internal release-review notes on
`backup`. It is excluded by [`.gitattributes`](../.gitattributes) from
`git archive` exports. When preparing a future clean `main` snapshot, export
the reviewed `backup` tree with `git archive` and check that the resulting
tree contains no `archive/` path. Do not merge all of `backup` into `main`
without that check. No change to `main` is made here.

- [`style-candidates/`](style-candidates/) contains the four white layout
  prototypes, their shared CSS, and the historical generator. Option 3 was
  selected for the live page; these source snapshots are not deployed by
  GitHub Pages.
- [`release-review/`](release-review/) contains the 5090 working-directory
  evaluation inventory, the detailed workspace provenance notes, and the
  inference/rights audit. Their conclusions remain available for team review.

## Former front-page results and reproducibility notes

The [online demo](https://zhanh-he.github.io/sfproxy-velocity-estimation/)
compares the original recording, Flat 64, Diff-Synth, and Diff-SFProxy on the
same notes and listening window. These selected examples are **not** aggregate
evaluations. Guitar datasets lack verified note-velocity labels; the paper
therefore evaluates rendered loudness correlations rather than guitar velocity
MAE. See the [listening-case inventory](../demo/analysis/LISTENING_CASES.md)
for excerpt sources and scores.

The reported tables are transcribed from the camera-ready paper. Recovered
5 s guitar models support the listening examples, but the exact run behind
the paper's **FL Diff-SFProxy r_BSSL = 0.777** remains unidentified. The
[recovery record](../demo/analysis/GUITAR_RESULT_RECOVERY.md) and
[workspace provenance](release-review/PROVENANCE.md) separate verified files
from unresolved release items. Do not treat an example checkpoint as a
reproduction of the full paper table.

To rebuild the paper-result graphics or serve the listening site locally:

```bash
python3 demo/analysis/build_figures.py
python3 -m http.server 8000 --directory demo/docs
```

Open `http://localhost:8000/`. The website is published from `backup`.
The [release-readiness audit](release-review/RELEASE_READINESS.md) records the
clean-environment inference check, guitar-weight provenance, FL result gap,
and component-specific rights evidence.
