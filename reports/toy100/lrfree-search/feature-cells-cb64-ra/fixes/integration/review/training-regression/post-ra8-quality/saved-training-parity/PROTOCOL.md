# Saved RA8 versus RA7 training neutrality

Freeze this descriptive helper before loading any RA8 numerical checkpoint.
Read only CPU copies of the ten original saved updates:
0,100,250,500,750,1000,1250,1500,1750,2000. Each candidate checkpoint must have
the post-save `training_checkpoint` JSON event in its existing training log.
Hash files before and after load. No CUDA, sampling, model construction,
optimizer steps, training, streams restored/consumed, or evaluator emissions.

Compare every serialized `trainer` leaf, with exact tensor dtype/shape/bytes
(including NaN and signed-zero bits), scalar types/values, dict keys and list
structure. Report per-component identities and concrete unexpected leaves.
Checkpoint `state_dict` contains FAST training weights even while serving EMA.
Also compare `data_position` and saved step. The outer record contains changed
serving metrics and log diagnostics, and its config receipt identifies a new
variant; these are reported as outer log/provenance, not training state.

The complete allowlist is:

- Backend schema6 ->7 and exactly two declared paired-average settings.
- The new typed backend `paired_average` stamp; its identical copy in last,
  and the new `last.paired_average_forward_rows` diagnostic.
- Existing `last.eval_seconds`, `last.work.distance_cells` and
  `last.work.projection_products` performance diagnostics.
- Cumulative `counters.feature_distance_cells` and
  `counters.projection_products` performance diagnostics.

All action/evidence/count/copy/birth counters and other work fields remain
compared. No rounding, tolerance or broad ignored subtree. Check the new
metadata's expected schema, policy, requirement, step, snapshot and duplicate
record consistency before removing it for legacy comparison. A difference
in a model, EMA, optimizer, controller, settler, row evidence, RNG, stream,
FIFO, graph or legacy reaction state is an actual reported divergence.

The watcher writes exclusive per-step receipts and immutable per-step seals.
Failed helper attempts remain separate. Its live index/log can grow until all
ten endpoints are compared. The source manifest pins both READY/package/config
and numerical lane source maps. The audit is training-neutrality evidence,
not quality acceptance, causal proof or historical intermediate-step replay.
