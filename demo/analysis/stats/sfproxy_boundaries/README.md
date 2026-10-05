This directory stores SoundFont-aware velocity boundary JSON files used by the
new SFProxy samplers.

Generate them with:

```bash
python diff-sfproxy/src/tools/discover_velocity_boundaries.py \
  --instrument_path /path/to/SalamanderGrandPianoV3.sfz \
  --backend sfizz \
  --instrument_name salamander_piano \
  --bank 0 \
  --program 0 \
  --sr 22050 \
  --seg_len_s 2.0 \
  --pitch_min 21 \
  --pitch_max 108 \
  --pitch_step 6 \
  --register_splits 48 72 \
  --hop 221 \
  --out_json demo/analysis/stats/sfproxy_boundaries/salamander_piano_boundaries.json
```

When a boundary JSON is absent, the sampler falls back to the default
`[0.33, 0.66]` boundaries.
