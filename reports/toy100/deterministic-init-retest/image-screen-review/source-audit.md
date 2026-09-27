# Fixed-initialization image harness review

Source and DV12 intensity2 CPU initialization PASS, bound to bundlef334d0038872de81dc2ac44d9b9a884c11bc66f9811945dd17f289f77d0b1ed5. All six complete model/template/scorer definitions and all four task specifications match the immutable archived image host. Models are residual-upsample width16/z8 with32 particles/batch32 and600 updates. Original templates,24 checks every25, HQ/mode bounds and final-five rule are unchanged.

The source retains CPU G→D→prior construction, actual public initializers, then CUDA transfer. The public trainer derives bandwidth/EMA/optimizer state afterward. Shared globalCUDA data/latent stream remains seed0, checked unchanged after constructors. A private clone computes all600 data/index/cursor receipts before updates; actual full serial publicsteps must match them. The same real batch feeds both losses. Evaluation enumerates all32 prior rows and uses the unchanged candidate generation path with isolated noise2303+completed, restoring training/caller state.

One independent guarded CPU check used the unchanged DV12 package on intensity2. It compared complete non-RNG state across two constructors with intervening random draws, verified exact R2 normalstd1 prior and initializer RNG neutrality, and checked device-local derived geometry/native empty state. Forward/backward/optimizer execution was forbidden and CUDA remained uninitialized. The runtime must also assert exact initialized CUDA model/buffer bytes against this CPU proof.

This is preparation only and clears only the declared DV12 intensity2 constructor proof. Other candidate/task executions still require matching receipts; no old image quality is inherited. Source/proof hashes and boundaries are in source-audit.json.
