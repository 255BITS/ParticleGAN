# External configuration recovery

Original launch records identify these configurations; no nearest-name substitution or old tensor loading is permitted. All source files remain unchanged. A recovered config does not clear its different probe interface for execution.

| Candidate | Binding status | Matching launch receipts |
|---|---|---|
| shared_coordinate | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| exposure_trust | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| exposure_mean | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| shared_geometry | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| visit_isotropic | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| density_mobility | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| optimistic_adam | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| optimistic_gradient | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| predictive_critic | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| joint_lookahead | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| extragradient | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| optimistic_critic | EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER | 2 |
| constraints_simple_regularization | PENDING_EXACT_BINDING_REVIEW | 0 |

The JSON preserves each original launch line hash, config/declaration/probe hash, and exact comparison against all200 canonical runtime files. Historical launch arguments using old CPU initialization fixtures are recorded for provenance only. The new retest must initialize every tensor afresh.
