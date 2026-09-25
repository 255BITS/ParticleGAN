Run a fresh GPU attempt from the selected direct-particle-response GAN:

```bash
./try-gan.sh --gpu 1 --minutes 45 --candidates 3 --workers 1
./try-gan.sh --gpu 1 --dry-run
./monitor-gan.py
```

The committed `reports/toy100/h_stability/SEARCH.md` defines the active task.
The active declaration is `reports/toy100/current-research-base.json`. It pins
`direct_particle_response`, including its config, mechanism, response and probe.
It has 15 PASS, 1 FAIL and 6 NOT_RUN on GPU. Gate on `vector_unequal_width` first,
then protect all fifteen passes. The config alone does not install this formula.
Full22 and own-state post-convergence stability remain unqualified.

Launch the next three-lane cycle:

```bash
python /ml2/hypergan/launch-gan-width.py --minutes 45 --proposals 3
```


The launcher uses a fresh gpt-6-astra/max session, isolated worktree, no memory
injection, and the existing Codex login. It runs on physical GPU1 by default,
requires CUDA, and clears inherited vendor preloads. `--gpu` selects a different
single GPU. Each attempt has a hard time cap and writes `tests.jsonl`, `result.md`,
`codex.log`, and status files. No automatic push or merge.

For three simultaneous attempts, give each `--workers 1` and a distinct `--focus`;
this keeps the total at three GPU benchmark processes. Candidate counts are
caps. Review measured results before starting another batch.

[Official non-interactive CLI documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
Previous CPU/constant-rate instructions are in `try-gan-continuous-history.md`.
