# Declared interpolation, not run

After the .50/.01 schedule and center-fixed β6 discriminator reached only a
three-check final passing streak, six nearby **global** hold/floor candidates
were declared in [beta6_interpolation_plan.json](beta6_interpolation_plan.json):

| Candidate | Cosine hold | Floor |
| --- | ---: | ---: |
| `sched_h45_f01` | .45 | .01 |
| `sched_h475_f01` | .475 | .01 |
| `sched_h525_f01` | .525 | .01 |
| `sched_h55_f01` | .55 | .01 |
| `sched_h50_f025` | .50 | .025 |
| `sched_h50_f05` | .50 | .05 |

The plan validated against the frozen rare-case host and exact β6
discriminator card, but **none of these six candidates was trained or scored**.
Work stopped before the first episode when an independent unchanged-c6
discriminator trial reported a sustained rare-case PASS and full 19-case
verification took priority. This plan has no index, receipts or behavioral
verdicts and contributes no selection points.
