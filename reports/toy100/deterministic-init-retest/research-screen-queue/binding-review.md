# Research constructor binding review

The exact17 retained installer functions/modules in `binding-exceptions.json` were inspected in full. Their dynamic aliases target only rate/noise functions (and the historical step-with-policy adapter). They do not replace G/D/prior constructors or the deterministic initializer. The candidate dynamic schedule/noise behavior remains unchanged. This clears a conservative constructor-routing flag only; it does not establish horizon independence, checkpoint completeness, or quality.

Each exception is accepted only for its exact function or complete module hash. Full inspected bodies are retained in `reviewed-binding-functions/`. All candidates still require independent construction/source receipts, and no GPU/training was executed in this review.
