# Current model/configuration scores

One configuration slot per family uses the same fresh common-26 comparison. NOT_RUN means accepted fresh comparison evidence is unavailable; eligibility is a separate status.

| Model/configuration | Representation | Fresh common-26 score/status |
| --- | --- | --- |
| R1/R2<br>[r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| BCap<br>[bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| K3P<br>[k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| KA2<br>[ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| E22<br>[e22](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| Full Atlas · [configs/100gaussians/atlas.json@a3ee5c67ac65](../../configs/100gaussians/atlas.json)<br>Configuration freeze pending | Particles (declared) | NOT_RUN (26 required)<br>[Eligibility: BLOCKED](technique-inventory.json#common26_display) |
| GAN v3 release 0.7 (MoG)<br>[release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| GAN v3 release 0.7 (cloud)<br>[release07-gan-v3-cloud-v1](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| K3P without critic anchor<br>[forge-onboarding-anchor-ablation](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| K3P without critic penalty<br>[forge-no-critic-penalty](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| K3P without A2<br>[k3p-a2-off-native-diagnostic](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |
| K3P without training output noise<br>[k3p-no-output-noise-diagnostic](../../configs/forge/selections/common26-display-audit-v1.json) | MoG / Particles (declared per task) | NOT_RUN (26 required)<br>[Eligibility: UNKNOWN](technique-inventory.json#common26_display) |

Historical and diagnostic evidence has separate configurations, serving laws and scopes. These links supply no fresh common-26 score:

- [Original Atlas recipe and serving-law evidence](continuous-baseline-20261003/README.md)
- [Original-suite stopped retest](pr223-original-full-retest-stopped17-20261004/README.md)
- [Native continuation first attempt](pr223-native3-first-invalid-20261004/README.md)
- [Repaired native continuation](pr223-native3-repaired-20261004/README.md)
- [C6 changed-rate / output-noise-off diagnostic](atlas-current-gpu-diagnostics-native-v2-20261003/README.md)
- [atlas_conditional adaptation evidence](atlas-named-gpu-diagnostics-native-v3-20261003/README.md)
- [atlas_routed adaptation evidence](atlas-named-gpu-diagnostics-native-v4b-20261004/README.md)
- [Retained word context](atlas-word-retained-context-20261004/README.md)
- [Word half-base rate contrast](word-half-base-20261004/README.md)
- [C6 baseline selection and retained diagnosis](c6-baseline-debug-20261003/README.md)
- [Separate completed study: critic_balance](critic-balance-20261003/README.md)
- [Separate completed study: generator_step](generator-step-20261003/README.md)
- [Standalone API evidence: k3p · 1-D Gaussian: histogram matching](../toy_audit/api_contract/gaussian1d/README.md)
- [Standalone API evidence: k3p · 1-D Gaussian: histogram matching · actual-training GIF](../toy_audit/api_contract/gaussian1d/goal.gif)
- [Differing canonical Atlas reference; not the requested Full Atlas configuration](../../configs/forge/ideas/atlas.json)

[Current declared view and additional-task scope](technique-inventory.json#declared_view): these separate declarations do not fill the fresh common-26 comparison.

## Qualification and scope

Fresh comparison evidence must bind the chosen configuration, unchanged required tasks, gates, budgets, source, runtime and serving law to a recognized new initialization and independent grade. Earlier source-bound records remain evidence in their own scopes.

Representation describes a declared prior or per-task host law. UNKNOWN means the chosen identity and complete prior contracts are not bound. The Full Atlas Particles label describes its requested original configuration; its common-task eligibility remains BLOCKED.

This display grants no current qualification, default adoption or speed comparison. The underlying scientific selection and historical records remain unchanged.

[Canonical rows, configuration alternatives and source-scoped evidence](technique-inventory.json) · [Evidence and archived publication identities](technique-evidence/manifest.json)

Regenerate this presentation from committed metadata:

```sh
python reports/forge/regenerate_technique_inventory.py --refresh-publication
```

Publication input digest `4993f3d5a8ba264d9adcf6b509cd02a34dc21243c2812974bdf219f4287834de`.
