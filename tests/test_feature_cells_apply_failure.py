"""A failed feature-cell evaluation keeps its accumulated evidence (review finding #21)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_feature_portability import make, owned_defaults, real  # noqa: E402,F401  (fixture reuse)

import particlegan.feature_cells as feature_cells  # noqa: E402


def test_failed_apply_restores_evidence_and_clocks_then_recovers(monkeypatch):
    trainer = make(backend='feature_cells')
    batch = real()
    captured = {}
    original_ready = feature_cells.FeatureCellBirthDeath.ready
    original_fit = feature_cells.FeatureCellSnapshot.fit

    def ready(self):
        result = original_ready(self)
        if result and self.snapshot_serial == 1 and not captured:
            # Distinct nonzero evidence so a silent zeroing is observable.
            self.S.fill_(.5); self.W.fill_(2.); self.n.fill_(3); self.pending[::2] = True
            captured.update(S=self.S.clone(), W=self.W.clone(), n=self.n.clone(),
                            pending=self.pending.clone(), serial=self.snapshot_serial,
                            evals=self.counters["evals"], rows=self.rows_since_eval)
        return result

    calls = {"fit": 0}

    def failing_fit(*args, **kwargs):
        calls["fit"] += 1
        if calls["fit"] == 2:
            raise RuntimeError("injected feature-cell failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(feature_cells.FeatureCellBirthDeath, "ready", ready)
    monkeypatch.setattr(feature_cells.FeatureCellSnapshot, "fit", staticmethod(failing_fit))
    with pytest.raises(RuntimeError, match="injected"):
        for step in range(40):
            trainer.step(batch + step * .001)
    bd = trainer.birth_death
    assert captured, "second evaluation never became ready"
    for key in ("S", "W", "n", "pending"):
        assert torch.equal(getattr(bd, key), captured[key]), key
    assert (bd.snapshot_serial, bd.counters["evals"], bd.rows_since_eval) == (
        captured["serial"], captured["evals"], captured["rows"])
    bd.maybe_apply(trainer, None)  # the retried evaluation succeeds and consumes the evidence
    assert bd.snapshot_serial == captured["serial"] + 1 and not bd.S.any()
    trainer.state_dict()
