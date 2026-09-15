# Handoff after restart: optimize CIFAR DDGAN training speed

Read scale_50k/READOUT.md, NEXT_ROUND.md and moonshot/READOUT.md first.

Current state: no active training or queued experiment. Host restart interrupted
NCSN++128 after last logged step27,200; last checkpoint/evaluation20k, diagnostic
FID65.209 (5k samples), versus240.594 at10k. This is not a completed50k run.
U-Net32 completed50k: final FID26.680 (50k samples),99.34 training minutes.
The no-argument default remains the fast10k U-Net32/pretrained-D recipe31.741.
All task code/config/reports are being committed as the fresh-session baseline.

Next user priority: speed up training while preserving the train_denoising.py
formulation. Profile first; no automatic continuation or new quality trial.
Both RTX A6000 GPUs authorized; GPU1 also drives the desktop. No seed-only runs.
Use full YAMLs, fresh output directories, one tail-friendly log, and report
throughput, memory, quality, and time spent evaluating separately. User prefers
waiting for scheduled runs to complete over repeated progress inspection.

Binding formulation: four-step DDGAN clean prediction and existing posterior,
joint timestep/class UCD, learned20k x128 latent particles, Gaussian step noise,
Rp logistic, candidate-only bcap, unique-row VICReg, existing optimizer recipe.
Pretrained D is pixel critic plus frozen eval-mode ResNet18 stages1/2/3 at64px;
input derivatives must still flow through the feature network for G and bcap.
Do not detach candidate features to obtain a misleading speedup.

Architecture options: unet, flat_hybrid, ncsnpp. Factory lib/image_moonshots.py;
NCSN++ wrapper lib/image_ncsnpp.py, vendored source lib/ddgan_ncsnpp/.
Official source commit/adaptations/licenses recorded in UPSTREAM.md. Native
PyTorch FIR fallback replaces compiled upstream CUDA; optional class embedding
is added to the processed timestep embedding. No upstream training loss imported.

Configs: configs/cifar_ddgan/scale_50k and scale_50k_smoke, plus earlier moonshots.
Completed/interrupted reports: reports/cifar-ddgan/scale_50k. Results/checkpoints
and source.zip remain in ignored results/cifar_ddgan/scale_50k. Strict resume
requires saved config/source; do not overwrite interrupted run with a benchmark.
Tail: tail -F results/cifar_ddgan/live.log. Cached CIFAR, FID and ResNet18 weights.

Pre-launch validation:34 focused tests +4 CUDA replay tests +2 GPU100-step smokes.
Replay tests use deterministic test-only pooling substitutions; production is
not bitwise deterministic. See scale_50k/validation.txt and tests/test_cifar_resume.py.
User mentioned optimization subagent work, but no agent/profile/patch was located
after restart. Treat that as missing context, not completed optimization.

Unrelated .claude/ worktrees and sparse-ucd.log must be preserved and excluded
from task commits. Existing review worktrees are unrelated. Never clean/reset
these to make git status appear empty. No push requested in this handoff.
