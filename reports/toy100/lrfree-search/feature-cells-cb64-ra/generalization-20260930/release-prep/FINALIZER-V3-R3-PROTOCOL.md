# Historical completed RA16 source closure

The live repository advanced after the RA16 full suite and original CUDA replay
completed. The completed suite checks its actual source/test bytes before and
after execution. Commit b25de08cfbeef9fe8aa06522324e132c3e71ae0f retains those
exact bytes. This fresh helper binds every declared source/test SHA to a local
immutable byte snapshot reconstructed from that commit, while still checking
the completed suite source map against the immutable RA16 package. No quality
criterion, numerical gate, replay rule, source routing or pytest count changes.

The old v3 metadata failure and v3-r2 live-source mismatch snapshots remain
closed and archived. Qualification remains RA16 on the original PR155 f459
reference. The advanced current PR155 source needs its separate bridge and
actual latest suite/replay. Every numerical execution keeps its original label.
