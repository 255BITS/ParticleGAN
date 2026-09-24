# Active base: port the passing CPU recipe to GPU

Use `configs/toy100/constraints_simple_regularization.json`, the original
scheduled 22/22 CPU recipe. The user has prioritized getting this recipe working
on GPU before continuing the constant-rate optimizer search.

Read the [porting results and replay](../cpu-recipe-gpu-port/README.md) and
[current base declaration](../current-research-base.json). The original recipe
scores 16/22 on native CUDA, including all three native 100-mode gates. Fresh CPU
training passes all six GPU-failing hosts. CPU initialization alone recovers
four of those six on GPU; all-CPU random draws recover a different four.
Neither diagnostic has completed a full 22-toy run.

First gate on ring and unequal mass, then trajectory and the three image
regressions, then the complete 22. Use the CPU-initialized CUDA control as the
porting reference. Preserve the scheduled GAN recipe, including its original
auxiliary AE/token host terms. Prioritize stability after convergence. Do not
mix results from different controls or add their partial passes to an old total.

The [launcher brief](SEARCH.md) defines current work. Use three experiments at
a time at most, fixed seeds, frozen gates, and easy-to-tail logs. Old epsilon/H
and shared-column-RMS runners are historical continuous-rate experiments.

[Earlier START](START-continuous-history.md) preserves their replay instructions.
The old `current-base.json` serves those archived runners; the active declaration
is `../current-research-base.json`.
