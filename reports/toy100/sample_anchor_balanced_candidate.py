"""Pre-start anchors with capacity-balanced mass and double-particle spread.

Same network, one Adam step per player, pre-G joint fit, and rest on a
nonconverged fit. The only change is the output objective: balanced quotas
replace nearest-centroid surplus collapse, and a two-particle group matches
the real minibatch's residual principal variance.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch

from reports.toy100 import sample_anchor_candidate as base
from reports.toy100 import sample_anchor_rest_candidate as rest
from reports.toy100.sample_anchor_mass_balance import balanced_field, balanced_mm_step
from reports.toy100.sample_anchor_prestart_candidate import PreStartRecorder
from reports.toy100.sample_group_anchor import mst_groups

METHOD = 'pr84_sample_group_anchor_prestart_balanced_mass'


class BalancedRecorder(PreStartRecorder):
    def correct(self, optimizer):
        centers, grouping = mst_groups(self.real)
        real = self.real.detach()
        members = grouping['member_indices']

        def step(support, centers_arg):
            return balanced_mm_step(support, centers_arg, members, real)

        def score(support, centers_arg):
            row = balanced_field(support, centers_arg, members, real)
            row['gradient'] = []
            row['gradient_l2'] = 0.0
            row['coverage'] = row['total']
            row['precision'] = 0.0
            row['assignments'] = []
            row['nearest_group'] = []
            return row

        with patch.object(base, 'output_mm_step', step), \
             patch.object(base, 'field', score), \
             patch.object(rest, 'field', score):
            super().correct(optimizer)
        row = self.corrections[-1]
        row['balanced_counts'] = row['mm'].get('counts')
        row['balanced_offset_norms'] = row['mm'].get('offset_norms')
        if row['step'] % 50 == 0 or row['step'] <= 3:
            print(json.dumps(dict(
                event='BALANCED_UPDATE', step=row['step'], selected=row['selected'],
                counts=row['balanced_counts'],
                max_offset=max(row['balanced_offset_norms'] or [0.0]),
                fit=row['fit']['status'])), flush=True)

    def receipt(self):
        result = super().receipt()
        result.update(
            method=METHOD, scratch_optimizer_policy=METHOD,
            added_objective=('capacity-balanced distinct-group targets; '
                             'two-particle groups match residual principal variance of the current real minibatch'),
            mass_rule='group counts differ by at most one; no nearest-centroid surplus collapse',
            shape_rule='doubles sit at centroid ± data principal axis, scale sqrt(max(0, emp var - 0.029^2)), cap 2*sigma',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def sample_anchor_balanced_candidate(*, task='mode_hold', start_step=0, correction=True):
    with patch.object(rest, 'RestOnFailureRecorder', BalancedRecorder):
        with rest.sample_anchor_rest_candidate(task=task, start_step=start_step, correction=correction) as value:
            yield value
