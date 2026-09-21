# Lunar Lander world-model replay

Open `index.html` locally; no server or dependencies required. The viewer replays fixed checkpoint outputs, with one-step and recursive modes. It does not run a live simulator or a learned policy.

The three-generator model uses `G1 -> st`, `G2 -> at`, `G3 -> st+1` from a shared learned MoG1024 draw. `E(st,at,terrain) -> z_hat -> G3` supplies the conditional prediction path. Every model receives privileged terrain context. Numerical results appear in the viewer.
