"""Source bytes shared by policy declaration, execution and grading paths.

The roster is metadata only. Each family's contract also binds its own
producer and original host; a common manifest grants no scientific credit.
"""

DECLARATION_SOURCES = frozenset({
    "experiments/forge/adapters.py", "experiments/forge/api.py",
    "experiments/forge/boundaries.py", "experiments/forge/contracts.py",
    "experiments/forge/mechanisms.py", "experiments/forge/planning.py",
    "experiments/forge/sampling.py", "experiments/forge/technique_board.py",
    "experiments/forge/views.py", "experiments/forge/runtime.py",
    "experiments/forge/evaluate.py", "experiments/forge/policy_adapters.py",
    "experiments/forge/policy_behavior_adapters.py",
    "experiments/forge/policy_snapshot_publication.py",
    "experiments/forge/policy_cohorts.py",
    "experiments/forge/named_policy_planning.py",
    "experiments/forge/policy_declaration_sources.py",
})
