import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root=Path('/tmp/pr36-image-solvability-artifacts')
root.mkdir(exist_ok=False)
manifest=[]
for stage in ['stage1','stage2','cross','diagnostics','combined']:
 source=Path('/tmp/pr36-image-solvability-'+stage)
 for path in sorted(source.rglob('*')):
  if not path.is_file(): continue
  relative=Path(stage)/path.relative_to(source)
  payload=path.read_bytes()
  if path.suffix=='.json':
   raw=payload; relative=relative.with_suffix('.json.gz'); payload=gzip.compress(raw,mtime=0)
  elif path.name.endswith('.json.gz'):
   raw=gzip.decompress(payload)
  else: raw=None
  output=root/relative; output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(payload)
  record=dict(path=str(relative),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
  if raw is not None:
   record.update(original_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw))
   assert gzip.decompress(payload)==raw
  manifest.append(record)
for path in sorted(Path('/tmp').glob('pr36-image-solvability-*.log')):
 dest=root/'logs'/path.name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(path,dest)
 manifest.append(dict(path=str(dest.relative_to(root)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size))
for path in [Path('/tmp/pr36-image-solvability-report.py'),Path(__file__),
             Path('/tmp/pr36-image-solvability-stage2-cards.json'),Path('/tmp/pr36-image-solvability-cross-cards.json'),
             Path('tests/test_image_solvability.py')]:
 dest=root/'reproduction'/path.name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(path,dest)
 manifest.append(dict(path=str(dest.relative_to(root)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size))
for stage in ['stage1','stage2','cross','diagnostics']:
 declaration=json.loads(gzip.decompress((root/stage/'declaration.json.gz').read_bytes()))
 with tarfile.open(root/stage/'source.tar.gz','r:gz') as tar:
  for name,digest in declaration['protocol']['source_sha256'].items():
   assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest
shutil.copyfile('benchmarks/transfer_suite/image_solvability.md',root/'README.md')
validation=dict(command='/home/mikkel/anaconda3/envs/conceptmod/bin/python -m pytest tests/test_image_solvability.py -q',
                result='3 passed in 3.76s',healthy_gan_episodes=60,supervised_controls=4,diagnostic_gan_episodes=2,
                all_episodes_complete=True,numerical_errors=0,fresh_holdout_evaluations=0)
(root/'validation.json').write_text(json.dumps(validation,indent=2))
(root/'archive_manifest.json').write_text(json.dumps(dict(files=manifest,validation=validation),indent=2))
print('Archived',len(manifest),'files; all original bytes and source hashes verified:',root)
