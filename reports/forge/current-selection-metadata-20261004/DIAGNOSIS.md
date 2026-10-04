# Feature selection ambiguity

The failure is a correct refusal of two unpinned metadata runtimes for the new draft `atlas-c6-observed-policy-current-v1`; existing selections were not overwritten.

At feature commit `37b4f05b697a529b88b9a02a1a324bbeb9da185e`, that idea has parent/preset Atlas and cohort `policy_selected_cloud_v1`, but neither registered family membership nor `trainer_family`. `family_for_candidate` therefore assigns a new singleton family (trainer_families.py:126–145). Missing-candidate discovery resolves CPU and CUDA rows (generator:1434–1452), each with zero attempts, 8 BLOCKED and 18 UNKNOWN of 26. The second runtime reaches the strict whole-row ambiguity guard at trainer_families.py:364.

The selection file remains SHA `a15b4a04d7159f1414ea510742a8471f4a3300c9b86b0201502c2e51b68853f1`, with all 12 original incumbent pins. It and the trainer-family registry are byte-identical to base `221f1c70447e7934f1aaf56ae30164c39f6e0aec`. The new idea is absent at that base. No measured row or numerical verdict was created.

The smallest proposed metadata correction is an explicit `trainer_family: "atlas"` on this prospective Atlas idea. An in-memory pure call confirms that classification. The existing exact Atlas incumbent then remains selected; the two draft runtimes are alternatives without new score or speed credit. If the idea is intended as a genuinely new family instead, it needs an explicit whole-row selection; the guard should remain strict.

The capture aborted before publication and did not run models, draws, official scorers, GPU work, raw hydration or Git mutations. Exact CPU/CUDA row, source, runtime and scientific hashes are in [diagnosis.json](diagnosis.json) and [prewrite.json](prewrite.json). No inference about root's active integration checkout was made from an obsolete worktree. Concurrent root edits are outside the committed capture.
