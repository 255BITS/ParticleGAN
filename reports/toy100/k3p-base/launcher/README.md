# Search engines

Install these launcher files together in the workspace root (locally,
`/ml2/hypergan`), alongside the selected research checkout and focused
`launch-gan-k3p-continuous.py` entrypoint.

The one-attempt launcher, formulation batch launcher and pool accept
`--engine codex|claude|grok|opencode`. OpenCode defaults to
`nano-gpt/meta/muse-spark-1.3-contributor`. The other engine defaults are unchanged.

```bash
# Inspect an attempt without creating a worktree or calling the model.
./try-gan.sh --engine opencode --dry-run

# After the search is resumed, launch one bounded attempt.
./try-gan.sh --engine opencode --gpu 1 --minutes 45 --candidates 3 --workers 1

# Inspect all K3P lanes using OpenCode.
python3 launch-gan-k3p-continuous.py --engine opencode --dry-run

# Inspect a pool using the focused K3P lane definitions.
python3 run-gan-pool.py --engine opencode --slots 3 \
  --lanes-from ./launch-gan-k3p-continuous.py --dry-run
```

`gan-attempts/STOP` blocks real attempts, including attempts started by a batch
or pool. Help and dry-runs remain available while paused. Adding this engine
does not resume the paused search.

OpenCode must be on `PATH`, or supplied via `OPENCODE_BIN`. Tested with OpenCode
1.18.31. Authentication uses its existing `nano-gpt` login first, then
`NANO_GPT_API_KEY` (with `NANOGPT_API_KEY` accepted as an alias). Configure a login
with `opencode auth login --provider nano-gpt` if needed. Keys are never written
to launcher artifacts. `--model` accepts OpenCode's full `provider/model` ID;
`meta/...` is also accepted as shorthand for `nano-gpt/meta/...`.

The runtime config uses NanoGPT's API host and registers the requested model
with a 1,048,576-token context and a conservative 32,768-token per-response
output cap. Reasoning uses the provider default. Each attempt gets a fresh
OpenCode session, an isolated config directory, shell/edit permissions, and
disabled delegation, interactive questions, web tools and external skills.
Project config and external plugins are disabled. Sessions remain in OpenCode's
local data store; sharing is disabled. The attempt timeout controls runtime;
`--budget-usd` remains Claude-only.

OpenCode artifacts are `opencode.log` (readable with `tail -f`),
`opencode-stream.jsonl` (raw events), `opencode-usage.json` (reported costs,
tokens and tool counts), `opencode-config.json` (no credentials), and `final.md`.
The existing `result.md`, `tests.jsonl`, patch and status contracts also apply.
The dashboard recognizes OpenCode logs and reported costs. Provider-reported
costs are estimates, not an enforced spending limit. An agent exit of zero
still does not qualify a research result.

Run the offline integration checks from the repository root:

```bash
python3 -m unittest discover -s reports/toy100/k3p-base/launcher -v
```

These use a temporary Git repository, fake CLI and fake GPU preflight. They
exercise prompt/model routing, credential-free artifacts, stream rendering,
error exits and the STOP guard without running training or calling a model.

References: [NanoGPT's OpenCode setup](https://docs.nano-gpt.com/integrations/opencode),
[OpenCode CLI](https://opencode.ai/docs/cli/), and
[OpenCode configuration](https://opencode.ai/docs/config/).
