PR331 showed that weak-singular-direction normalization amplifies tiny Ring16
gradient differences at update 401. This experiment asks whether adding
float32-rounding-scale noise in those directions changes acquisition, either
once after a fresh live 400 prefix or continuously from initialization.

The experiment-scoped helper adds `U_weak @ N @ Vh_weak`, with independent
normal entries of scale `float32_epsilon*sigma_max/sqrt(k)`, before the original
polar update. It uses one fixed numerical-rank threshold, constant learning
rates and a separately checkpointed CUDA noise stream. Biases, prior updates,
public initialization, data draws, batch and full numerical bounds are fixed.
Candidates never load the archived checkpoint.

No neural experiment ran: CUDA is unavailable on this host. Syntax, bound
declarations, planning and refusal to start CPU training/probing pass. The
frozen allowance is 3,200 updates/600 training seconds plus one 30-second
saved-gradient check, with no retries. Actual-training GIFs and numerical
candidate outcomes remain pending CUDA execution; GitHub publication remains
pending network access.

The saved-gradient probe checkpoints cloned perturbation streams and bounds
actual gradient perturbation, without assuming noise improves sensitivity or
convergence. Production defaults and Forge qualifications remain unchanged.
See `reports/forge/ring16-noise/README.md` for exact formulas, evidence identities,
budgets, commands and comparison limits.
