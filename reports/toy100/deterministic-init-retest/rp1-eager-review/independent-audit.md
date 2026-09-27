# RP1 historical eager-state diagnostic review

PASS for source preparation and CPU constructor-only integration, under the explicit historical worker diagnostic label. This does not turn the setup into ordinary public API behavior.

The candidate package and complete recipe match the already reviewed RP1 public-partial port. The added five-line optimizer-state loop is AST-identical to the immutable historical worker. The separate harness changes only its post-constructor setup and setup receipt; original host, objectives, sampling, evaluator and sealed v1 harness are unchanged. Both added helpers are source archived.

The allowlisted helper refuses nonempty optimizer state or an advanced trainer, verifies its source hash, creates only the original zero Adam states, and asserts every counter/moment's shape/dtype/device/value. Both CPU constructions retained exactly9G/8D states and identical complete nonoptimizer state/RNG before and after setup. Full initial model parameters/buffers and all nonoptimizer learner state equal the public-partial CPU proof; repeated builds after extra RNG draws are identical. No forward/backward/optimizer step or CUDA initialization occurred.

[Hash-bound independent audit](independent-audit.json) · [CPU preflight](cpu-preflight.json) · [Both setup witnesses](setup-receipts.json). Actual CUDA sampling/device validation remains mandatory before external updates.
