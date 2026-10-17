"""Build a readable index from completed saved-result and export receipts only."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def build():
    directory = HERE / 'publication'
    data = json.loads((directory / 'results.json').read_text())
    rows = {r['attempt_id']: r for r in data['certified_results']}
    assert len(rows) == 148 and len(data['media']) == 143
    lines = ['# Actual-training media', '',
        'Each GIF illustrates certified saved training evidence. Fifteen 100-Gaussian GIFs use saved generated/target snapshots; five mode-hold GIFs show actual stored training measurements, with no saved generated samples. The five clock diagnostics retain state proofs and are outside this 143-GIF denominator.', '',
        'All exports add zero optimizer updates, sampling draws or qualification credit. Ordinary and diagnostic outcomes remain separate. [Exact metrics and export/source receipts](results.json).', '',
        '| Family | Scope | Task | Recorded grade | Saved illustration | Export receipt |',
        '| --- | --- | --- | --- | --- | --- |']
    for media in sorted(data['media'], key=lambda m: (m['family'], m['scope'], m['task_id'])):
        row = rows[media['attempt_id']]
        gif = directory / 'media' / Path(media['path']).name
        receipt = gif.with_suffix('.json')
        assert hashlib.sha256(gif.read_bytes()).hexdigest() == media['sha256']
        assert hashlib.sha256(receipt.read_bytes()).hexdigest() == media['receipt_sha256']
        assert media['receipt']['optimizer_updates_added'] == media['receipt']['sampling_draws_added'] == 0
        lines.append(f"| {media['family']} | {media['scope']} | {media['task_id']} | {row['gate_status']} | [GIF](media/{gif.name}) | [receipt](media/{receipt.name}) |")
    (directory / 'MEDIA.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(gifs=len(data['media']), attempts=len(rows), source_digest=data['source_digest'], paid_seconds=data['total_paid_seconds'])))

if __name__ == '__main__':
    build()
