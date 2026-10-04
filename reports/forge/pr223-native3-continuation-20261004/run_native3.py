"""Root-only native3 metadata envelope; scientific execution stays run_retest.py.

Importing this file does not inspect devices, create a ledger, prepare a source,
register work, construct a model, or read any old scientific artifact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = 'reports/forge/pr223-native3-continuation-20261004'
HELPER = 'reports/forge/pr223-original-full-retest-20261004/run_retest.py'


def helper():
    name = '_pr223_native3_execution_helper'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT/HELPER)
    loaded = importlib.util.module_from_spec(spec);sys.modules[name]=loaded
    spec.loader.exec_module(loaded)
    return loaded


def metadata_plan(closed_anchor, *, root=ROOT, output=None):
    """Root invocation only: charge the SAME old ledger; never creates/reset it."""
    runtime = helper();scope = runtime.native3_module()
    ledger = runtime.ledger_module().SharedMetadataLedger(scope.require_existing_ledger())
    with ledger.phase('native3_source_only_plan'):
        packet = scope.plan(root,runtime,closed_anchor)
        scope.validate_live_history(packet,ledger.snapshot(),runtime)
        if output is not None:
            path = Path(output).resolve()
            if path.exists():
                raise ValueError('new declaration output already exists')
            runtime.atomic_json(path,packet)
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-metadata', type=Path)
    parser.add_argument('--closed-metadata-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--queue-root', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--plan-file', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--copied-preflight', type=Path)
    parser.add_argument('--max-new-attempts', type=int)
    args = parser.parse_args(argv)
    runtime = helper();scope = runtime.native3_module()
    if sum(bool(x) for x in (args.plan,args.prepare_only,args.copied_preflight))>1:
        parser.error('choose exactly one metadata operation or run')
    if args.copied_preflight:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='': parser.error('copied preflight must hide CUDA')
        prepared = runtime.read_json(args.copied_preflight)
        if prepared.get('schema')!=scope.SCHEMA: parser.error('exact native3 prepared packet required')
        result = runtime.bounded_copied_preflight(prepared)
        print(json.dumps(result,sort_keys=True));return 0
    if args.closed_metadata is None or args.closed_metadata_sha256 is None:
        parser.error('root SHA-bound closed parent metadata copy required')
    if args.closed_metadata_sha256!=scope.ANCHOR_SHA:
        parser.error('closed metadata copy must match committed stopped17 FINAL_COST')
    # Only pin names are assembled here; byte/source validation is inside the
    # charged operation below. This prevents unaccounted plan/source work.
    closed = dict(path=str(args.closed_metadata.resolve()),sha256=args.closed_metadata_sha256,bytes=scope.ANCHOR_BYTES)
    if args.plan:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='': parser.error('source-only plan must hide CUDA')
        packet = metadata_plan(closed,output=args.plan_file)
        print(json.dumps(dict(status=packet['status'],required=3,original_catalog_required=19,
            active_case_ids=list(scope.IDS),prior_case_charged_seconds=scope.PRIOR_CASE_SECONDS,
            case_caps_sum_seconds=4290,maximum_inclusive_campaign_seconds=scope.PRIOR_CASE_SECONDS+4290+180,
            metadata_ledger_path=scope.CANONICAL_LEDGER,source=packet['source']['commit']),sort_keys=True))
        return 0
    if args.output is None: parser.error('--output required for prepare/run')
    if args.max_new_attempts is not None and args.max_new_attempts not in (1,2,3):
        parser.error('dispatch bound must be 1..3; no other case admission')
    if args.prepare_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='': parser.error('source preparation must hide CUDA')
        packet = runtime.prepare(args.output,queue_root=args.queue_root,native3_anchor=closed)
    else:
        for key,value in runtime.protocol.ENVIRONMENT.items():
            if key=='CUDA_VISIBLE_DEVICES' and os.environ.get(key) not in (None,'1'):
                parser.error('physical GPU1 numeric placement required')
            os.environ[key]=value
        packet = runtime.run(args.output,queue_root=args.queue_root,native3_anchor=closed,
                             max_new_attempts=args.max_new_attempts)
    print(json.dumps(dict(status=packet['status'],required=3,original_catalog_required=19,
                         completed=packet.get('completed',0),new_paid_seconds=packet.get('new_paid_seconds',0),
                         prior_case_charged_seconds=scope.PRIOR_CASE_SECONDS,
                         waiting_reason=packet.get('waiting_reason')),sort_keys=True))
    return 0 if args.prepare_only or packet['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
