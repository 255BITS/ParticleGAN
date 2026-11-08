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


def test_failure_after_mean_phase_keeps_checkpoint_valid_and_resets_moved_rows(monkeypatch):
    trainer = make(backend='feature_cells')
    batch = real()
    original_phase = feature_cells.run_mean_phase
    calls = {"n": 0}
    seen = {}

    def failing_phase(backend, trainer, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            seen["serial"] = backend.snapshot_serial - 1
            trainer.prior.z.data[5] += 1.  # a row the failed body moved before moved_rows is assigned
            raise RuntimeError("injected mean-phase failure")
        return original_phase(backend, trainer, *args, **kwargs)

    def ready(self):
        result = original_ready(self)
        if result and calls["n"] == 1:
            self.S.fill_(.5); self.W.fill_(2.); self.n.fill_(3)  # distinct evidence to be restored
        return result

    original_ready = feature_cells.FeatureCellBirthDeath.ready
    monkeypatch.setattr(feature_cells.FeatureCellBirthDeath, "ready", ready)
    monkeypatch.setattr(feature_cells, "run_mean_phase", failing_phase)
    with pytest.raises(RuntimeError, match="injected"):
        for step in range(60):
            trainer.step(batch + step * .001)
    bd = trainer.birth_death
    assert calls["n"] == 2
    # The mean phase had already counted an evaluation; counters roll back with the serial.
    assert bd.snapshot_serial == seen["serial"] == bd.counters["mean_evals"]
    make(backend='feature_cells').load_state_dict(trainer.state_dict())  # previously rejected
    others = torch.arange(len(bd.n)) != 5
    assert bd.n[5] == 0 and bd.W[5] == 0  # the moved row starts a new lineage
    assert bool((bd.n[others] == 3).all()) and bool((bd.W[others] == 2.).all())
