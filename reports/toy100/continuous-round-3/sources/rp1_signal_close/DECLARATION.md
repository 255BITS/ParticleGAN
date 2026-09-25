# rp1_signal_close

Parent: pinned K3P. `config.json` `a1475108…`, `latent.py` `197df635…`, and `response.py` `7e71d60a…` are byte copies. Learned particle prior, direct response, and bounded sparse-latent damping are unchanged.

Diagnostic predecessor, not a base: AP3 held 1200/1200 and extension 300/300, pre-hold 120/120, recovery FAIL 72/81. Eight modes returned by step 2600 while critic-gradient level was already under 0.25, then nine deadline checks failed at steps 3070–3170 while rate gain was still forced at 0.2 (`refit_until` 3201). The peak reset at +200 inflated that level, and quiet never reached 250, so the rate never decayed.

## Rules

Cold path matches AP3. Critic-gradient RMS versus a peak set after 200 steps and decayed at 0.9997. Full rate and the early penalty until 250 consecutive steps below a quarter of that peak, then rate gain and mixing gain decay at 0.99. When mixing gain first hits 0 it stays 0: capped real/fake gradients plus the 0.999 EMA anchor.

A later reopen (level at least 0.5, or fast/slow RMS at least 3 for 3 steps) sets rate gain to 0.2 and leaves the mixing weight at 0. There is no 800-step hold and no peak reset. After 250 quiet steps the rate decays at 0.99. Another reopen can raise it to 0.2 again. Applied network multiplier `0.01 + 0.99 * rate_gain`. Applied prior multiplier `0.05 + 0.95 * rate_gain`. Guard remains 5× Adam RMS after 200 steps.

## Noise

Input noise 0.5 to 0 over 120 updates. Output noise 0 to 0.029 over 240 updates. These are declared step counts, equal to K3P's fractions of a 1200-update horizon, and they do not read `noise_horizon` or the training budget. A zero output-warmup fraction would stay constant; K3P's config warmup is 0.2, so the 240-step ramp is the one used.

## Not read

Total training budget, evaluation scores, convergence detection, target identities, shift time, target centers, critic cosine. Host step and horizon arguments to the rate multiplier are ignored.
