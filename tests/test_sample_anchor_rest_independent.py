"""Source-bound independent checks of the sample-anchor nonconvergence guard."""

import os
from pathlib import Path

import pytest
import torch

from reports.toy100.pr84_sample_anchor_rest_independent_audit import (
    forced_favorable_native, forced_one, success_parity,
)


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / 'reports/toy100/continuous-evidence'
BASE = EVIDENCE / 'round6-sample-anchor-independent/saved44'
REST = EVIDENCE / 'round6-sample-anchor-rest-independent/saved44'


def root():
    path = Path(os.environ.get('SAMPLE_ANCHOR_REST_ROOT', REPO)).resolve()
    if not (path / 'reports/toy100/sample_anchor_rest_candidate.py').is_file():
        pytest.skip('sample-anchor rest source has not been integrated in this checkout')
    torch.set_num_threads(1)
    return path


def test_frozen_success_path_is_bitwise_equal():
    result = success_parity(BASE, REST)
    assert result['compared_updates'] == 88
    assert result['guarded_success_updates'] == 44


def test_forced_nonconvergence_restores_live_g_and_prior():
    result = forced_one(root(), REST)
    assert result['forced_network_and_prior_perturbations']
    assert result['all_g_prior_parameters_exactly_restored_to_pre_g']
    assert result['d_both_adam_ema_rng_noise_identical_to_original']


def test_forced_nonconvergence_overrides_beneficial_native_fallback(tmp_path):
    capture = os.environ.get('SAMPLE_ANCHOR_REST_CAPTURE')
    if capture is None:
        pytest.skip('original selected-state capture not available')
    result = forced_favorable_native(root(), REST, Path(capture), tmp_path / 'source')
    assert result['natural_native_improves']
    assert result['forced_fit_nonconverged_rested']
    assert result['all_g_prior_parameters_exactly_pre_g']
