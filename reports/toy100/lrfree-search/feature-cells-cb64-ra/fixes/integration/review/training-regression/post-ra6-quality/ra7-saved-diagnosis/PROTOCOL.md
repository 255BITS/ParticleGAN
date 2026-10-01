# RA7 saved milestone diagnosis

Run CPU read-only analyses at sealed500,1000 and final2000 checkpoints only. A seal requires the checkpoint file and its postwrite training_checkpoint event. The independent state watcher owns state validity and population-clock analysis.

Reuse the unchanged frozen analyze_saved.py and compare_checkpoint_motion.py from post-ra5-saved-diagnosis, passing RA7 run/package/READY paths. Output goes into fresh milestone subdirectories here. The framework's historical RA5 labels do not identify the input package; pinned RA7 source/READY maps do.

The existing framework computes current clean FAST/EMA outputs, saved emitted metrics, learned-parent accessibility and latest birth-row fitness, and includes matched RA4/E22 saved controls. A standard-library summary joins the frozen RA6 receipt by checkpoint step without rerunning its numerical analysis. Serving and official emitted metrics remain unchanged.

Fixed-coordinate comparisons use250→500,500→1000 and1000→2000, respectively. These compare saved generators and saved latent tables, with old CPU critic/head geometry held fixed for the old-coordinate cohort. They do not reconstruct historical GPU partitions or birth incarnation lifetimes. No optimizer virtuals, training, new data, new seeds, proposals, emissions, CUDA or RNG advancement.

RA7's three rate changes quarter G and learned log-sigma optimizer bases while preserving prior and D base rates. The noise formula, floor, mode, evaluator, seed, data and horizons are unchanged. This scope is explicit; there is no posthoc EMA or noise override.

At each milestone preserve scripts' source/input maps, captured metric prefixes, outputs, logs and interpretation in an independent freeze manifest. Prior frozen RA6/RA7 sources and evidence remain read-only. Intermediate outcomes do not qualify final strict toy and canonical full-grid gates.
