import gzip,hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path('/ml2/hypergan/ParticleGAN-transfer-vectors')
base=Path('/tmp/pr36-transfer-vectors-20260922')
out=Path('/tmp/pr36-transfer-vectors-handoff-2062458')
out.mkdir(exist_ok=False)
index={'commit':'20624589e10da9bc670136cb46409a2da953fe23','base_commit':'6c499e6ce5d05af3673c4636142c39f8cc0ffb37','note':'Original JSON bytes are preserved by gzip. v2 rescores existing metrics and did not retrain. Reserved data was never sampled or evaluated.','json_files':{},'source_archives':{},'files':{}}
tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
paths=[p for p in tracked if p.endswith('.py') or p in ('pyproject.toml','requirements.txt','AGENTS.md','benchmarks/transfer_suite/vector_tasks.md')]
for version in ('v1','v2'):
    hashes={}
    destination=out/(version+'_source.tar.gz')
    with tarfile.open(destination,'w:gz') as archive:
        for name in paths:
            old=base/'v1_source'/name
            src=old if version=='v1' and old.exists() else root/name
            data=src.read_bytes()
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(name)
            info.size=len(data)
            info.mode=0o644
            info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    index['source_archives'][destination.name]={'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'source_sha256':hashes}
for src in sorted(base.rglob('*.json')):
    if 'source_snapshot' in src.parts or 'v1_source' in src.parts:
        continue
    name=str(src.relative_to(base))
    dest=out/'results'/(name+'.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    data=src.read_bytes()
    dest.write_bytes(gzip.compress(data,mtime=0))
    index['json_files'][str(dest.relative_to(out))]={'original_relative_path':name,'original_bytes':len(data),'original_sha256':hashlib.sha256(data).hexdigest(),'gzip_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
for src in sorted(base.glob('*.py'))+sorted(base.glob('*.log'))+[base/'v2_rescored/README.md']:
    dest=out/'support'/src.relative_to(base)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(src.read_bytes())
    index['files'][str(dest.relative_to(out))]=hashlib.sha256(dest.read_bytes()).hexdigest()
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
readme=['# Vector transfer calibration handoff','','Code commit: `20624589e10da9bc670136cb46409a2da953fe23`. Exactly 8 development tasks plus an unexamined reserved family. All 12 recorded reference executions used seed0, CPU, one Torch thread.','','- [Protocol v2 readable leaderboard](support/v2_rescored/README.md)','- [Original v1 source archive](v1_source.tar.gz)','- [Corrected v2 source archive](v2_source.tar.gz)','- [Original-byte hashes and archive inventory](index.json)','- [Tailable original log](support/progress.log)','- [Original calibration script](support/calibrate.py)','- [Additional fixed references](support/extra_references.py)','- [Explicit v2 rescoring script](support/rescore_v2.py)','','All JSON files are gzip-compressed without changing their uncompressed bytes. `index.json` maps relative paths to original SHA256 and compressed SHA256. Original v1 source includes the original module/tests/design document; v2 contains the corrected code. Both archives include tracked Python dependencies from the isolated worktree. No reserved samples or results are present.','','V2 adds the per-component minimum eigenvalue bound to every separated mixture after a review witness demonstrated a loophole, before controller fitting. Original executions are retained and v2 rows explicitly distinguish execution from scoring protocol. No training was repeated for this correction.','','## JSON files','']
for name in index['json_files']:
    readme.append(f'- [{name}]({name})')
(out/'README.md').write_text('\n'.join(readme)+'\n')
print(out)
print('JSON files:',len(index['json_files']))
print('Bytes:',sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
