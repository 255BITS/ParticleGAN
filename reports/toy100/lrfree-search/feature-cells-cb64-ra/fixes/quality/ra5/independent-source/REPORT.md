# RA5 independent composition source audit

PASS. The package contains exactly 29 Python modules. Its canonical byte digest
is `7c5067ca3071d5e345d38317f4f28dc984d8e596254148a64e8c53eaf5b1b270`.
The config is byte identical to RA4 (`d2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7`).
All 194 reviewed input/source hashes remained unchanged.

The independent script explicitly reverses every root integration splice. This
reconstructs the complete frozen PAIR `feature_cells.py` **bytes**, including
decorators. Reversing the frozen PAIR `_move`, schema and policy additions then
reconstructs the complete RA4 module AST. There is no undeclared module delta.

- Twenty-four original modules are byte exact from PAIR/RA4.
- `continuous.py` and `training.py` are byte exact from the population owner.
  Its frozen manifest digest and separately declared canonical byte digest are
  each verified under their stated conventions.
- `_move` is the exact decorated PAIR method. It retains the shared noise draw
  and computes live/EMA displacements on their own prior geometry before writes.
  All three count planner methods remain byte exact. The original count family,
  certificates, topology, precision/noise laws and bounded geometry are retained.
- `select_parents` is the exact decorated BIRTH contract method. The two new
  helper modules match both the owner files and its frozen contract package.
  Root hooks share the ordinary action budget/ledger, reserve novel births and
  seeds, apply paired live/EMA coordinates, and include all copied or born child
  rows in the existing tester/row-evidence rebase hook.
- Trainer schema 5, backend schema 6, copy noise policy and novel birth policy
  describe the distinct composed law. State/action/replay mechanics remain the
  subject of the owner and root CPU contracts; this audit repeats none of them.
- The indexed `_generate` method is byte exact from frozen RA4. New birth
  decisions use observed real features, learned heads and prior geometry. They
  contain no benchmark labels, task branches, held-out score calls or file data
  access. The canonical harness retains its original SHA and quality gates.

Two precomposition guard defects were reported and repaired by root before
composition: owner relative/full package maps now verify before copying, and
the new config path refuses an overwrite. No prior frozen evidence was edited.

This audit uses only stdlib source/AST/hash operations. It imports no candidate
or Torch, creates no CUDA context and runs no numerical job, training update or
seed experiment. Strict learned toy and grid quality are pending; the source
PASS is independent of those eventual quality verdicts.

The complete proof, file maps and owner READY identities are in `receipt.json`;
`audit_source.py` contains the explicit inverse proof.
