# Actual CUDA-default portability regression

The root launches exactly the frozen public portability test node against the
immutable RA16 package, before the latest full suite. The worker holds the
inherited existing shared serial lock and verifies both original parked PID
identities. Only physical GPU0 is visible, its UUID is checked, memory fraction
is .2, deterministic algorithms are enabled, and TF32 is disabled.

The test repeats the same fixed seed1234 and explicit CUDA model, prior and
data recipe under ambient CPU and CUDA device contexts. Each context performs
18 updates through feature reactions, restores a valid checkpoint, compares
the next update on both original and restored trainers, and compares samples.
There are40 actual diagnostic step calls across the two contexts. The shared
elapsed clock is diagnostic only. No quality scorer or threshold is changed.

Source preparation and check-only verification use stdlib and initialize no
CUDA runtime. The root alone launches GPU work. Fresh logs and receipts retain
both success and failure; no process is signalled by this launcher.
