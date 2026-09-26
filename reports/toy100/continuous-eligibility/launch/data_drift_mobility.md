# Lane 2: distinguish changed data from the learner's own instability

Own a controller with separate evidence for real-data change and generator/game
instability. The old DI2 real-mean latch opened mobility and nearly recovered,
then lost a mode after its real-data score returned to normal. It cannot see
generator-side forgetting or changes preserving the mean. DI3's anchor snap
worsened recovery. KA2 surprise alone also fires on stationary collapse.

Declare a general, normalized detector using ordinary minibatches or learned
features that does not consult benchmark target centers, labels, change times,
quality scores or task IDs. If using real-data statistics, they may control
mobility; do not directly fit/translate generator outputs to those statistics.
Provide a separate stability signal for self-induced drift and cold acquisition.
Show autonomous closing and reopening without DI2's inherited K3P decay/noise
schedule or a forced known recovery dwell. Keep controller state checkpointed.
Other lanes own PX3-like reference-motion control and a fixed-rate repair.

Read-only lead:
`reports/toy100/continuous-round-3/completed-data-innovation/attempts/k3p_data_innovation/result.md`
in the supplied research evidence checkout. Its old 81/81 and seed language is
superseded by COMMON.md. Do not rerun DI1/DI2/DI3 unchanged.
