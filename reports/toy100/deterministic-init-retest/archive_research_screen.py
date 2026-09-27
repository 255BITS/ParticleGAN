"""Preserve a reviewed research-host screen separately from public API scores."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from archive_screen import summarize

ROOT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()


def render_table(table):
    lines = ['# Research-host results with the new initialization', '',
        'These runs preserve their original research learner and use the reviewed new public initializer. They remain separate from public API qualification. Every score requires all eight modes and HQ≥90% for the final five of 24 observations.', '',
        'A quality pass does not establish continuous-learning eligibility. [The closeout](closeout.md) records the current eligibility decisions, promising SN3 result, incomplete qualification and user-requested stop. [Exact source audits](research-eligibility-audits/) preserve each configuration’s separate limitations.', '',
        '| Research configuration | Result | Passing observations | First arrival | Final streak | Final modes / quality |',
        '|---|---|---:|---:|---:|---|']
    for row in sorted(table['results'], key=lambda row: (row['status'] != 'PASS', -row['summary']['passing'], row['candidate'])):
        s = row['summary']
        lines.append(f"| [{row['candidate']}]({row['archive_manifest']}) | {row['status']} | {s['passing']}/24 | {s['first_arrival'] or '—'} | {s['final_suffix']} | {s['final_modes']}/8 / {s['final_hq']:.1%} |")
    lines += ['', 'Inner host diagnostic labels do not override the strict eight-mode score. Original schedules, source limitations, and old-initialization evidence remain attached to each configuration.', '']
    (ROOT / 'research-leaderboard.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--eligibility', default='REQUIRES_OWN_CONTINUOUS_USE_EVIDENCE')
    args = parser.parse_args()
    assert args.id.replace('-', '').isalnum()
    audit = json.loads(args.audit.read_text())
    assert audit['status'] == 'PASS'
    source = Path(audit['source'])
    for name, digest in audit['artifacts'].items():
        assert sha((source / name).read_bytes()) == digest, name
    result = json.loads((source / 'result.json').read_text())
    receipt = json.loads((source / 'initialization-receipt.json').read_text())
    rows = result['result']['observations']
    assert [row['step'] for row in rows] == list(range(50, 1201, 50))
    summary = summarize(rows)
    assert result['status'] == audit['quality_status'] == ('PASS' if summary['final_suffix'] >= 5 else 'FAIL')
    assert not receipt['historical_tensor_fixture_loaded'] and not receipt['old22_scores_inherited']
    table_path = ROOT / 'research-results.json'
    table = json.loads(table_path.read_text()) if table_path.exists() else dict(schema=1,
        scope='RESEARCH_HOST; results do not establish public GANTrainer qualification', results=[])
    assert args.id not in {row['id'] for row in table['results']}
    destination = ROOT / 'research-evidence' / args.id
    destination.mkdir(parents=True, exist_ok=False)
    artifacts, checkpoints = {}, {}
    for name in audit['artifacts']:
        path = source / name
        raw = path.read_bytes()
        if path.suffix == '.pt':
            checkpoints[name] = dict(path=str(path), sha256=sha(raw), bytes=len(raw))
            continue
        target = destination / (name + '.gz' if len(raw) > 100000 and path.suffix in ('.json', '.jsonl') else name)
        target.write_bytes(gzip.compress(raw, mtime=0) if target.suffix == '.gz' else raw)
        artifacts[str(target.relative_to(ROOT))] = dict(sha256=sha(target.read_bytes()),
                                                       original_sha256=sha(raw), bytes=target.stat().st_size)
    entry = dict(id=args.id, candidate=receipt['candidate'], scope='RESEARCH_HOST',
        status=result['status'], summary=summary, seconds=result['seconds'],
        continuous_eligibility=args.eligibility, source_directory=str(source),
        audit=str(args.audit.resolve()), audit_sha256=sha(args.audit.read_bytes()),
        artifacts=artifacts, raw_checkpoints=checkpoints, old_quality_inherited=False)
    manifest = destination / 'archive-manifest.json'
    manifest.write_text(json.dumps(entry, indent=2) + '\n')
    entry['archive_manifest'] = str(manifest.relative_to(ROOT))
    table['results'].append(entry)
    table['recorded_utc'] = datetime.now(timezone.utc).isoformat()
    table_path.write_text(json.dumps(table, indent=2) + '\n')
    render_table(table)
    print(json.dumps(dict(id=args.id, status=result['status'], summary=summary)))


if __name__ == '__main__':
    main()
