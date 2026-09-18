# Completed scout review and promotion

All three runs passed source/config/summary certification. Initial model state
and training RNG streams matched. Each ran alone on GPU 0, with only the bcap
interval changed. All scores below use 50,000 generated images.

| N | FID50k ↓ | Steps/s | Training minutes | Total minutes |
|---:|---:|---:|---:|---:|
| 8 | 21.993 | 22.02 | 3.78 | 5.28 |
| 4 | 22.276 | 19.42 | 4.29 | 5.78 |
| 16 | 23.930 | 23.89 | 3.49 | 4.99 |

Promote **N=8**: 13.4% higher throughput than N=4 and a small 0.283 FID advantage.
The quality gap is too small to establish a general advantage from one trajectory;
the useful result is the measured speed gain with no observed degradation here.
N=16 buys only another 8.5% throughput versus N=8 while worsening FID by 1.937.
Its slightly lower reconstruction MSE (.08395 versus .08558) does not outweigh
the generation-quality loss for this goal.

Sample grids for N=8 and N=16 contain varied recognizable CIFAR-like objects,
with distortions still visible. At N=8, encoder usage is 263 effective particles;
33.8% of offset coordinates are near the bound. These diagnostics do not establish
unconditional mode coverage, and remain worth watching during longer training.

N=8 runtime projections: **25.4 minutes for 30k**, **41.8 for 50k**, and
**82.7 for 100k** updates, including scheduled evaluation. A final FID50k
generation/evaluation costs about **71 seconds**. These are constant-throughput
projections, not forecasts of reaching a particular FID. See the benchmark
limitations and sources in [LEADERBOARD.md](LEADERBOARD.md).

The next run uses 30k updates, identical initialization/optimizer settings,
FID5k every 5k updates, final FID50k, all 10k test reconstructions, and numbered
checkpoints. It starts a fresh trajectory with the same seed; checkpoint
continuation is not implemented. A final read-only FID5k audit enables a
same-count comparison with earlier evaluations.

```sh
bash experiments/cifar_ae_lazy_long.sh
tail -F runs/cifar_particle_ae/lazy_long/PIPELINE.log
```
