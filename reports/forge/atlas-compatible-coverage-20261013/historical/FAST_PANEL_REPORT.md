The faithful current-API adapter is mechanically verified, and its fresh fast panel is complete: **7 primary clean passes,4 failures across11 fixed cases**. Six vector cases and Stripes pass their sustained gates. Bars, Blobs and Intensity fail; ring16 passes its endpoint but only4 terminal checks, below the unchanged5-check requirement. This is an opt-in adapted profile, not a generic preserves-all-wins default.

|Case|Actual/original updates|Primary sustained verdict|Passing suffix/observations|First observed pass|Current-B context|
|---|---:|---|---:|---:|---|
|vector_unequal_width|1200/1200|PASS|20/24|250|FAIL|
|vector_anisotropic|1200/1200|PASS|20/24|250|FAIL|
|vector_two_broad|1200/1200|PASS|23/24|100|PASS|
|vector_unequal_mass|1200/1200|PASS|20/24|250|FAIL|
|vector_overlap|1200/1200|PASS|23/24|100|FAIL|
|vector_spiral|1600/1600|PASS|23/24|134|PASS|
|img_stripes2|600/600|PASS|14/24|275|PASS|
|img_intensity2|600/600|FAIL|0/24|—|PASS|
|img_bars4|600/600|FAIL|0/24|—|FAIL|
|img_blobs4|300/600|FAIL|0/12|—|PASS|
|ring16_acquisition|400/400|FAIL|4/24|350|UNMEASURED|

The six existing outstanding tasks were frozen from the certified current-B request, with exact target/scorer/gates/cadences: four vector cases,1200 updates except spiral1600, plus two600-update image cases. One global archived RA14 recipe is retained. No seed/config search, winner selection, parameter tuning or target/gate change was performed. The complete fast roster is measured; no long qualification or independent holdout was launched.

The vector constructor preserves rawParticlePrior20000×2, affine2×2 G, nativeD128×3/Fourier3, batch2048, full reg3/KA2/DV12/legacy complete-group semantics and original output-sigma binding. Public seed0 initializer uses G0/D1/prior2; caller CPU real streams D100/G101 and trainer streams1/2/3/4 are isolated. Changing the external constructor cap1200 to400 for ring or1600 for spiral introduces no recipe schedule; recipe.total_steps remainsNone. Preflight verifies the cap-only initial full-state identity for ring. Task target means/covariances define data/scoring and never enter a controller or loss as labels.

Images retain the frozen convolutional G/D architecture and exact32-row indexed evaluation. Their explicit dimensional adaptation is z2->8, table20000->32, batch2048->32. FAST clean is primary. Vector primary means the public state-selected full-policy path, including DV12 and serving averages. EMA, served image views, and paired original output-noise cohorts have separate conclusions. The actual output_sigma floor is preserved rather than substituted with a latent MoG width. No clipping or normalization enters scoring.

Ring finishes with16 quality modes, HQ .969727, massTV .020996, full component covariance error .350312 and minimum eigenratio .626650. All four views first pass at350 and retain4 passing checks at350/367/384/400. The operational job completed normally, but **the scientific sustained verdict is FAIL**. No extra check or extension was added to manufacture a fifth pass.

The new vector results preserve meaningful shape/allocation: two-broad clean covariance error .060936, minimum eigen .970199; unequal-mass .088692/.813223, massTV .007520. Overlap has SW .020765, mean error .019944 and global covariance error .025342. Spiral has SW .017257, mean error .004286 and covariance error .061632. Every clean/noisy selected/EMA cohort passes in these four cases. Required component gates include every nearest-center assigned row; no core/tail trimming replaces them. Inherited task particles256 only affects separately named ungated resolved diagnostics. It does not describe the actual20k table and grants no finite-atom exemption.

Stripes acquires two orientations from275 and sustains14 checks, HQ1.0, clean template RMSE .000514 and TV .0625. Every image view passes. Intensity completes600 with0/24 passes, HQ0, zero quality modes and mean template RMSE .175000; it fails photometric acquisition. This task uses the unchanged stricter quality RMSE .06. Bars completes600 with0/24; Blobs stops300 under its predeclared stall rule with0/12 and remains **incomplete under its original600-update task**. Blobs nearly blacks out; Bars becomes a saturated wrong pattern. Neither noisy/EMA serving repairs those failures.

The source's32-row KNN fallback cannot act through its isolation bound (minimum conformal p1/17>.05). Bars/Blobs have excess death requests without certified deficit parents, therefore zero transport moves. Stripes' pass is counterevidence against a blanket image incapacity claim; backend inactivity alone does not establish the cause of the failed cases. The automatic KNN G/noise rate .00425 also differs from the vector feature-cell rate .0010625. This is a documented coupled architecture/population/backend difference, not a separately tested causal intervention.

Task12's distinct matched reg.5 KA2 and reg.5 symmetricBCAP cases all retain width/aniso passes. Thus strength/family alone do not reproduce current-B collapse on the preserved successor host. Its BCAP implementation is an explicitly bound external frozen kernel, because public5737 does not expose that recipe family; resuming it requires its own source envelope. Task13's prior/initializer attribution and Task14's local contraction/noise probes remain separate. Neither their costs nor their results are folded into this panel.

The original current-B context has5 positive short cases; this adapted profile retains3 (two-broad,spiral,stripes) and loses2 (Blobs,intensity), while passing4 context failures (width,aniso,unequal-mass,overlap). This locates preservation gaps. Different architecture/prior/population/source/streams prevent a matched superiority claim or filling current-B/MoG qualification cells with these scores.

The new saved-array audit reproduces768 scorer dictionaries exactly and decodes14 initial/final metadata checkpoints, with zero model construction/forward/update. Earlier216 image and192 vector cloud audits remain hash-bound: all1176 stored cohorts in this panel are audited. All252 original observation boundaries assert complete trainer-state purity; caller data RNG/cursors and training streams are checkpointed. Actual-training GIFs use saved clouds only. The read-only independent review confirms frozen task/source contracts and the full-component versus ungated-mask distinction.

The native adapter's previous90 real resumed updates across3 layouts remain numerically exact for learned state,losses and samples. Its original literal strict-state failures are preserved: only write-only eval_seconds metadata differed; raw9/30 strict matches are not rewritten to30/30. The separate full historical replay and original clean/noisy qualifications are unchanged. Historical .029 noisy wins grant no current clean or MoG credit.

All9 new scientific jobs completed under normal admission; the old width/aniso raw post-training JSON-export ERROR receipts remain ERROR, with their separately audited report recovery. The image metadata-audit list/dictionary error is retained and charged; completed Blobs figures were reused. New admitted work since the earlier vector handoff paid841.844636940s, with901.844636940s conservative charges. Prior758.956952464s is retained, producing cumulative1660.801589405s, within the explicitly declared4000s cap. CPU preflights/import errors are recorded separately; peer lane costs are excluded. All own workers are settled.

Keep the archived qualified profile and the mechanically verified adapter frozen. The complete fast panel does not support a generic preserves-all-wins release, a ring extension, an independent holdout or long qualification. No source default was edited, original artifact rewritten, merge or publication performed.

[Mechanical successor](/ml2/hypergan/atlas-ra14-compatibility-20261010/SUCCESSOR_HANDOFF.json) · [Previous width/aniso report](/ml2/hypergan/atlas-ra14-transfer-panel-task11-20261010/TRANSFER_REPORT.md) · [Image failure report](/ml2/hypergan/atlas-ra14-image-fast-task11-20261010/IMAGE_FAST_REPORT.md) · [Current saved audit](/ml2/hypergan/atlas-ra14-fast-roster-task11-20261010/saved-audit-v1/admission/228757195f2d259d/results/RESULT.json) · [Independent contract review](/ml2/codex-storage/future-workspaces/2026-10-09/task-11/case-discovery/FAST_HANDOFF_REVIEW.md).
