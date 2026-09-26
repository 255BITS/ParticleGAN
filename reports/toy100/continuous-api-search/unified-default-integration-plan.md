# One release/API default after qualification

Prepared from the unchanged PR195 API checkout at fa511ce0. This is a change
map, not a selected candidate or implemented promotion. PR195 targets develop;
PR155 retains research evidence. Neither is merged.

The winner must be the same resolved policy in Recipe(), get_recipe(),
GANTrainer and public custom-loop components. A finite example/evaluation run
may stop after a caller-chosen number of updates without making that number an
input to rates, noise or controller state.

| Existing surface | Concrete integration requirement |
|---|---|
| particlegan/recipes.py Recipe.total_steps defaults7000 | A continuous default must not require a learner end. Pick one qualified policy; no hidden different winner for direct Recipe construction. |
| get_recipe family names gan/mog/ddgan/AE etc | Names configure model families. Preserve that contract and verify the selected controller with conditional/encoder/prior bindings; do not silently fall back to a different optimizer for unsupported shapes. |
| training.py step and load_state_dict use recipe.total_steps as upper bound | Continuous stepping and checkpoint resume must work beyond prior observation endpoints. Retain strict recipe/controller/state compatibility checks and scoped autograd execution. |
| input/output noise helpers depend on total_steps | Install exactly the qualified horizon-free noise policy. Initialization windows may remain only as actually declared and tested. |
| public LR/component factories | Bind all stateful scales and accepted-update clocks. RP5 needs precision plus joint transaction; DV7 needs its additional critic factor. A component loop that omits one is not the tested learner. |
| README.md examples | Replace range(recipe.total_steps) and get_recipe(total_steps=example_budget) with separate caller run limits. Explain simple target streaming and automatic adaptation. |
| examples/quickstart_gan.py | --steps/--stop-after must limit execution only. Resume should extend run length without changing recipe/controller trajectories. |
| examples/pytorch_loop.py | Same controller/transaction through public components, not manual reproduction of internal equations. |
| examples/100gaussians.py | _RECIPE.total_steps //1000 would fail for None. Keep epochs/logging limits in example configuration; remove their influence on learner schedules. |
| examples/five_modes.py and examples/api.toml | Same resolved default and caller-only observation/run limits. Explicit experimental overrides must not claim default qualification. |
| package/version/checkpoint metadata | Distinguish the newly qualified state schema from finite KA2/K3P checkpoints. Document actual supported migration; do not silently reinterpret stored optimizer/controller state. |
| historical benchmark runners | Keep labeled historical research evidence. New qualification must use the selected public implementation with frozen task losses and observations. |

After selection, integrate in the existing develop-targeted draft, run the
repository-required API/serialization/build checks and meaningful public-loop
parity/continuation checks. Recheck any changed arithmetic with its own quality
measurements. The PR body should say why the chosen candidate satisfies the
user's continuous-learning requirement, cite its own measured scores and state
remaining limits. Keep unmerged for user review; no release publication.
