# RA8 saved Grid100 covariance

Evidence VALID; original Grid100 FAIL. All five terminal checks fail. At7000, original precision .96995<.97, maximum covariance1.819186>1.7, center RMS .20357>.2, radial KS .04120>.04. The independent holdout passes the separate accuracy limits but fails the frozen coverage/shape gate. No gate or source changes.

## Excess direction

Mode69, center(1.5,4.5), carries the largest conditional eigenvalue in both original sample sizes. Terminal20k direction118.43 degrees:1.819186 = clean1.417993 + saved output-noise .782145 + conditional cross−.380952. Holdout100k direction120.09 degrees:1.730715 = clean1.271083 + noise .833828 −.374196. Components use the same noisy nearest-mode IDs and same noisy radius<=3 subset; covariance identities close below1e-10. Original float32 GPU maximum is reproduced to2.2e-16 by the fixed diagnostic equations.

The global saved output-noise variance is .938133 and .929422 target sigma squared, close to the prescribed(.029/.03)^2=.934444. The conditional covariance difference is concentrated in anchor/clean spread; this is not evidence of an enlarged output-noise scale. Other holdout high modes are80(max1.5400) and8(max1.4380). Boundary modes have only modestly greater mean max-eigenvalue1.1480 versus interior1.0956, so the failure is not a universal edge effect.

## Raw anchors, perturbation and copy links

Mode69 raw FAST/EMA anchor max-eigenvalues are1.57524/1.47276; the saved served clean holdout gives1.52010. Thus broad raw anchors already account for most of the clean excess. Saved noisy and clean arrays labeled live and EMA are exactly identical under the positive average lease; they are not independent FAST-versus-EMA cloud controls. The clean channel includes the existing latent perturbation. Original sampled row identities were not saved, so marginal anchor-versus-clean comparisons cannot identify each row's jitter or establish its causal contribution.

Holdout mode69 has9.865% noisy radius>3 tails; its raw FAST/EMA anchor tail rates are1.717%/.858%. Across all holdout tails, mean radial energy15.7006 decomposes into clean7.0525 + noise5.8016 + cross2.8465. Only16.66% of noisy tail samples already have clean component radius>3;22.12% have noise component radius>3. Most tails arise from their joint displacement. These fractions overlap and are not an additive causal attribution.

Known-copy graph:10605 undirected edges,9395 components, largest26 rows, mean degree1.0605. Four edges cross an oracle mode. In worst mode69, largest known component is5.58% of its233 anchors and mean degree1.2704. This bounded local graph does not demonstrate one dominant copy family; discarded links prevent a full genealogy or effective sample size claim.

## Learned feature chart

The historical chart reports64 cells, rank8 and35 real-only topology groups for100 annotated modes. Its exact projection/centers/group mapping are deliberately absent from the checkpoint and were not refitted with new RNG. The current saved-D head inputs, standardized on even FIFO references before projection, still separate real mode centroids by at least10.25 own within-mode RMS. Every raw FAST/EMA anchor is nearest the feature centroid of its annotated mode. Coarse cells/topology, rather than observed raw feature collapse, are therefore the supported aliasing concern. This descriptive metric is not the historical support/count partition or a distribution certificate.

The final empirical paired lease is positive19113/19000 with20000 same-group rows; it does not certify these within-mode moments. A separately frozen bounded-neighbor probe will test the radius/width hypothesis on a small fixed subset. No candidate correction is qualified by these diagnostics.

## Scope

Helper/protocol and63 source/input identities were frozen before measurement. CPU only, no random draws, training, optimizer steps, new quality emission, chart refit, source or saved artifact mutation. Original results and strict FAIL remain authoritative. Full per-mode covariance matrices/eigenvectors/lineage associations and all five failure rows are in result.json.
