# Saved-training media publication

[`export_media.py`](export_media.py) publishes one representative actual-training
GIF for every task with a terminal executed result in the completed inventory
campaign, including newly eligible Tier 2 tasks. It preserves the recorded
grade and uses retained scored samples, word views, numeric trajectories or
clock-control states through the existing Forge renderer.

Selection is fixed before loading arrays: choose the exact selected BCAP
dualnorm configuration when it executed that task; otherwise choose the
lexicographically first candidate, revision and attempt identity. Grades do not
enter selection. Invalid, nonfinite or absent selected evidence produces an
explicit unavailable entry with its original grade and reason. The exporter
does not substitute another candidate. Required tasks without an executed result
remain listed separately; diagnostic tasks do not enter that required count.

Run from the executed source checkout after the complete campaign drain. Source
commits come from saved requests. The renderer must match the file hash in each
request's frozen source manifest. The default campaign is
`gaussian-smoke-inventory-v2`; pass paths for the actual queue and attempt archive.
The output directory must be new.

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/gaussian-smoke-inventory/export_media.py \
  --queue-root runs/forge/gaussian-smoke-inventory-v2 \
  --attempts reports/forge/attempts \
  --output reports/forge/gaussian-smoke-inventory/media \
  > runs/forge/gaussian-smoke-inventory-v2/media-publication.log 2>&1
tail -F runs/forge/gaussian-smoke-inventory-v2/media-publication.log
```

`selection.json` binds chosen identities and the queue snapshot before rendering.
`index.json` retains every selected gate, unavailable reason, source identity,
certificate hashes, saved-input hashes and GIF/renderer receipt hashes. Individual
GIF receipts identify the actual observation frames. These are presentation
artifacts with `qualification_input: false`; they do not rescore observations or
change the inventory leaderboard. Preserve the complete raw campaign archive
and its immutable digest independently of these compact files.

This process loads saved arrays on CPU for plotting. Neural execution remains
on GPU in the original attempts. Publication disables model construction and
forward calls, trainer updates and trainer/prior sampling in its own process.
It adds zero training updates and zero sampling draws. An active campaign is
rejected before creating an output directory.

Software checks covered grade-independent preferred selection, lexicographic
fallback, missing required Tier 2 coverage, exclusion of diagnostic coverage,
active-campaign refusal, nonfinite rejection and the execution guard. Seventeen
retained v2 attempt envelopes were also checked against their frozen source,
candidate and result certificates while the campaign ran, without rendering.
A separate compatibility check rendered nine frames from the preserved shallow
smoke observations with execution disabled: GIF SHA-256
`df05a5acefcad1a6cf3e86479793c2081693b6132263f771df9cf6137232b496`.
That check retains prefix source
`326f0e0b82d014dfcd728cd8f5719b2450fc627a` and original receipt SHA-256
`dd8aee7c637708be28b7f19b9eff506674ae1eaa0e6bb9ac281449b7c31665a8`;
it supplies software compatibility evidence, with no new campaign qualification
or new training.
