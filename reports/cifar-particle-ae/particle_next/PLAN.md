# Particle count and training duration experiments

User authorized both arms on 2026-09-18.

| GPU | Particles | Steps | FID50k evaluation | Question |
|---|---:|---|---|---|
| 0 | 16,384 | 40k → 80k | Every 10k | Does the recent improvement persist with more training? |
| 1 | 32,768 | 10k → 40k | Initial, then every 5k | Does another doubling improve the matched count curve? |

32k starts from the same original 1,024-particle 10k checkpoint used by the earlier count scouts. Duplicated centers preserve its initial distribution; optimizer/EMA moments, saved sigma and normalization reference remain matched. 16k resumes its existing 40k checkpoint. One discriminator update and the existing E-only reconstruction recipe are unchanged. No seed repeats.

Baseline: 16k at40k FID50k17.0982; 8k17.7683; 4k17.2350. Best previously observed checkpoint remains4k35k16.5033. Goal below13. More decodable sibling information has not consistently implied better density/coverage, so interpret all metrics together.

Each arm automatically runs read-only endpoint information and density/coverage probes, then stops. Compare32k at40k with prior counts at40k; compare16k at80k with its own duration curve. Review trends and selected best versus endpoint before any further count increase or200k promotion.

Validation: four factor32 trainer tests passed on GPU1, six information tests passed, actual8-update32k training smoke and32k information GPU smoke certified. New standalone trainer/probe preserve historical source certificates. Information probe adds independent seed bands for children16–31 while preserving previous first16 child draws. Training smoke FID128 is not a benchmark.

Logs: `tail -F runs/cifar_particle_ae/particle_next/PIPELINE.log`. Per-arm launcher logs in that directory expose pipeline failures. Process metadata in LAUNCH.json. Expected runtime roughly35–40 minutes including evaluations and endpoint probes, subject to measured throughput.
