# Circle toy moved to a fresh transition experiment

The active collaborator handoff is now:

[Circle transition-encoder start guide](https://github.com/255BITS/ParticleGAN/blob/experiment/circle-transition/docs/circle-toy.md)

Branch: `experiment/circle-transition`, based on PR #18's merged implementation.

The new experiment drops the explicit-memory requirement and all previous circle
runs as baselines. It uses current observed state/context through an encoder and
action head, with shuffled local training and full trajectories for frozen
evaluation only. Its circle leaderboard starts empty.

This `feat/sequential-memory-path` branch is an archive. Its code, reports and
checkpoints are not prerequisites for the new toy. Historical requirements,
leaderboards and next-experiment recommendations here do not govern the new work.
