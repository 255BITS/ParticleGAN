# Paired research workflow

`phase3.py` prepares one unchanged chosen baseline and one substantive global candidate on a common frozen source. The caller supplies the mechanism, baseline identity, research predictions and prior evidence. The helper contains no chosen baseline or research hypothesis. The [template](phase3-spec-template.json) is deliberately unfinished; a ready study rejects its `TODO` fields.

The input freezes all six original Tier 1 questions plus these ten Tier 2 questions: `gaussian1d_stability`, `five_word_joint_hold`, `trajectory`, `residual_student`, `vector_unequal_mass`, `vector_unequal_width`, `vector_anisotropic`, `grid100`, `rotated100`, and `staggered100`. Task IDs must resolve from the original ordinary revision-8 view. Original architecture, data law, prior, initialization, sampling, updates, schedule, cadence, dependencies and numerical gates remain fixed. Source rebinding touches only affected declarations in this selected roster, recording previous hashes and evaluator revisions; unrelated Gaussian shallow cards remain unchanged.

Each arm reserves **22,920 seconds** and the pair reserves **45,840 seconds**. The registered paid ceiling is **48,000 seconds**, with **24,000 seconds per arm** including any separately admitted execution repairs. Targeted software checks have a separate **300-second allowance** per track. The runner provides no automatic retries, extra configurations or tuning. These are full reservation ceilings, not runtime predictions.

Preparation creates a fresh `research_diagnostic` view with all sixteen assignments at tier 1 and importance `diagnostic`. Independent deeper jobs can run after another diagnostic fails. Gaussian stability and word hold still require that same candidate's completely passing original producer and exact own checkpoint. No diagnostic can borrow baseline states or acquire ordinary qualification.

Use a separate worktree and unique campaign, view, candidate, study and registration IDs for each track. Put the reviewed spec and registration inside that checkout. Candidate cards use schema 3; the existing public study schema remains version 1 with status `ready`, and Forge must compute admission `READY` for both arms. Commit the actual implementation and all source/declaration bytes before admission.

```sh
archive=/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/replace-track-id
mkdir -p "$archive/logs"
python reports/forge/bcap-three-phase/phase3.py --prepare \
  --spec reports/forge/replace-track-id/spec.json \
  --registration reports/forge/replace-track-id/registration.json \
  --source-commit "$(git rev-parse HEAD)" > "$archive/logs/prepare.log" 2>&1
# Review and commit the generated candidate/study/view, selected source pins,
# spec and registration. The next two calls must name that exact committed HEAD.
python reports/forge/bcap-three-phase/phase3.py --submit \
  --registration reports/forge/replace-track-id/registration.json \
  --artifacts "$archive" --source-commit "$(git rev-parse HEAD)" \
  > "$archive/logs/submit.log" 2>&1
python reports/forge/bcap-three-phase/phase3.py --drain \
  --registration reports/forge/replace-track-id/registration.json \
  --artifacts "$archive" --source-commit "$(git rev-parse HEAD)" --gpus 0,1 \
  > "$archive/logs/phase3-driver.log" 2>&1
tail -F "$archive/logs/phase3-driver.log" "$archive/phase3-queue/events.jsonl"
python reports/forge/bcap-three-phase/phase3.py --publish \
  --registration reports/forge/replace-track-id/registration.json \
  --artifacts "$archive" --source-commit "$(git rev-parse HEAD)" \
  --output "$archive/publication" > "$archive/logs/publish.log" 2>&1
```

`--prepare` resolves real preflight without creating a worker or queue. `--submit` freezes both requests and admits them without starting a worker. `--drain` requires both prior admissions, unchanged registration and exact HEAD. Execution and saved-evidence publication reject a changed source. Admission checks that the spec, registration, candidate cards, studies, view, tasks and scientific files are tracked and match the committed bytes.

Publication reuses the earlier saved publisher by adapting its module roles, scope and task selection at runtime; historical implementation and evidence remain unchanged. It validates certified attempts and source manifests, audits paired initialization, prior, named RNG and actual target-batch identity, and verifies each executed continuation's own producer. Different completed budgets or missing arm states remain explicitly unverified. The publisher exports actual-training GIFs from certified saved observations under a guard that forbids live training or sampling. It writes scoped scalar outcomes, research study predictions/falsifiers, blockers, costs, retry history, provenance and media receipts with `qualification_input=false` and zero added updates/draws.

The result is a scoped research readout, never a second generated goal leaderboard, 21/21 claim or default adoption. Preserve the repository's single current technique inventory. Bulk logs, JUnit, samples, checkpoints and GIFs stay in the external archive; commit compact metrics, provenance and a report linking the archived evidence after completion.
