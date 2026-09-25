from pathlib import Path
import gzip,hashlib,io,json,tarfile,shutil
root=Path('/ml2/hypergan/ParticleGAN-learned-lr')
dest=root/'reports/smart_descent/sgd'
phases=[('common_lr',Path('/tmp/pr36-sgd-study'),'sweep.json',['/tmp/pr36-sgd-sweep.log','/tmp/pr36-sgd-fit.log','/tmp/pr36-sgd-diagnostics.log']),('per_tensor',Path('/tmp/pr36-relative-sgd-study'),'sweep.json',['/tmp/pr36-relative-sgd.log']),('rms_decay',Path('/tmp/pr36-relative-sgd-followup'),'followup.json',['/tmp/pr36-relative-sgd-followup.log'])]
all_reports={}
for name,source,primary,logs in phases:
    out=dest/name
    out.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/primary).read_text())
    all_reports[name]=report
    manifest={'compression':'gzip.compress(payload,mtime=0)','original_bytes_preserved':True,'protocol_sources_verified':True,'artifacts':{},'source_bundle':{}}
    for path in sorted(source.rglob('*.json')):
        rel=path.relative_to(source)
        original=path.read_bytes()
        compressed=gzip.compress(original,mtime=0)
        target=out/Path(str(rel)+'.gz')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(compressed)
        assert gzip.decompress(compressed)==original
        manifest['artifacts'][str(rel)]={'original_sha256':hashlib.sha256(original).hexdigest(),'compressed_sha256':hashlib.sha256(compressed).hexdigest(),'original_bytes':len(original),'compressed_bytes':len(compressed)}
    hashes=report['protocol']['source_sha256']
    for path,digest in hashes.items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
    sources=set(hashes)
    sources.update(str(p.relative_to(root)) for p in (root/'benchmarks/learned_lr').glob('*sgd*.py'))
    sources.update(['benchmarks/learned_lr/__init__.py','benchmarks/learned_lr/controller.py','tests/test_sgd_controller.py'])
    bundle=io.BytesIO()
    with tarfile.open(fileobj=bundle,mode='w') as archive:
        for rel in sorted(sources):
            payload=(root/rel).read_bytes()
            info=tarfile.TarInfo(rel)
            info.size=len(payload);info.mtime=0;info.mode=0o644
            archive.addfile(info,io.BytesIO(payload))
            manifest['source_bundle'][rel]=hashlib.sha256(payload).hexdigest()
    raw=bundle.getvalue();compressed=gzip.compress(raw,mtime=0)
    (out/'sources.tar.gz').write_bytes(compressed)
    manifest['source_tar_sha256']=hashlib.sha256(raw).hexdigest()
    manifest['source_tar_gz_sha256']=hashlib.sha256(compressed).hexdigest()
    (out/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for logfile in logs:shutil.copy2(logfile,out/Path(logfile).name)
    if name=='common_lr':
        text=(source/'README.md').read_text().replace('.json','.json.gz')
        text+='\nDownload [exact source](sources.tar.gz) and [archive hashes](archive_manifest.json). All raw JSON downloads use deterministic gzip; the manifest records original-byte SHA256s.\n'
        (out/'README.md').write_text(text)
    else:
        rows=report['rows']
        if name=='per_tensor':
            title='# Per-tensor relative-step grid'
            intro='This is per-tensor adaptive gradient descent, not common-LR SGD or Adam. Each step is `−alpha_tensor × raw_gradient`, with no update momentum. The rule is `alpha = clip(target_role × max(parameter_RMS,.01) / max(gradient_RMS,1e-12), base_LR × 1e-4, base_LR × 100)`. Base rates are G=.25/D=.005. Targets are swept independently over .001/.003/.01/.03. The same ring4/grid9 development tasks, seed 0, and 1,200 updates are used throughout.'
            cols='| Config | G target | D target | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Sustained tasks |'
            sep='| --- | ---: | ---: | ---: | ---: | ---: | ---: |'
        else:
            title='# Per-tensor RMS memory and decay'
            intro='Targets remain G=.01/D=.01, selected by the preceding grid. This follow-up varies only target-fraction cosine decay and a causal scalar running RMS denominator. Running RMS uses an EMA of gradient-RMS squared initialized from the first gradient. It stores one scalar per tensor, includes the current gradient, and accumulates no update momentum or coordinate-wise Adam moments. Existing development tasks are reused; these are not fresh transfer results.'
            cols='| Config | RMS beta | Cosine start | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Sustained tasks |'
            sep='| --- | ---: | ---: | ---: | ---: | ---: | ---: |'
        lines=[title,'',intro,'','Live weights determine all scores; EMA is separate in the raw records. Sustained success requires full mode coverage, HQ≥90%, and at least five passing observations through the final step. The objective strongly rewards each sustained task, so one task can improve the mean while the other regresses. Overall success still requires both tasks.','',cols,sep]
        for row in sorted(rows,key=lambda x:x['objective']):
            a,b=row['episodes'];fmt=lambda e:f"{e['live'].get('modes','—')}/{e['live'].get('n_modes','—')} / {e['live'].get('hq',0):.2%}"
            stable=sum(e['convergence']['stable_from_step'] is not None for e in row['episodes'])
            x,y=(row['g_fraction'],row['d_fraction']) if name=='per_tensor' else (row['rms_beta'],row['cosine_start'])
            lines.append(f"| {row['name']} | {x if x is not None else 'current'} | {y if y is not None else 'none'} | {row['objective']:.4f} | {fmt(a)} | {fmt(b)} | {stable}/2 |")
        if name=='rms_decay':
            lines+=['',f"The constant/current-RMS control exactly reproduces the preceding grid: `{report['parity']}`."]
        lines+=['',f"Download [all scores and protocol]({primary}.gz), [exact source](sources.tar.gz), and [original-byte archive hashes](archive_manifest.json). Every feature/action/metric trace and separate EMA result is retained in `episodes/*.json.gz`.",'']
        (out/'README.md').write_text('\n'.join(lines))

common,relative,followup=(all_reports[k] for k in ['common_lr','per_tensor','rms_decay'])
rows=[('Adam / cosine',{'episodes':common['controls'],'objective':sum(x['objective'] for x in common['controls'])/2}),('Tuned common-LR SGD / constant',common['best_constant']),('Tuned common-LR SGD / cosine',common['best_cosine']),('Per-tensor current RMS / constant',relative['best']),('Per-tensor current RMS / cosine .6',next(r for r in followup['rows'] if r['name']=='current_cosine06')),('Per-tensor running RMS .99 / constant',next(r for r in followup['rows'] if r['name']=='rms099_constant'))]
lines=['# Raw-gradient descent research','',
'No tested raw-gradient rule sustained full coverage and HQ≥90% on both development tasks. Common-LR SGD collapsed; per-tensor steps substantially improved coverage, and target decay produced a sustained ring4 pass. Shared performance across tasks remains unresolved. No new distribution or architecture was used to tune this study.','',
'| Method | Ring4 live modes / HQ | Grid9 live modes / HQ | Sustained tasks | Objective ↓ |',
'| --- | ---: | ---: | ---: | ---: |']
for name,row in rows:
    a,b=row['episodes'];fmt=lambda e:f"{e['live'].get('modes','—')}/{e['live'].get('n_modes','—')} / {e['live'].get('hq',0):.2%}"
    stable=sum(e['convergence']['stable_from_step'] is not None for e in row['episodes'])
    lines.append(f"| {name} | {fmt(a)} | {fmt(b)} | {stable}/2 | {row['objective']:.4f} |")
lines+=['',
'The objective includes a 20-point penalty for each task without sustained success, then final/late coverage and HQ deficits and a small SW1 term. Its mean therefore ranks the single-task cosine success highly despite worse grid coverage. This is not an overall PASS. All rankings use live weights; raw artifacts report EMA separately.','',
'1. [Common-LR SGD and scalar feedback](common_lr/README.md): 16 G/D rate pairs under constant/cosine schedules, 64 episodes; 36 numerical failures. A 16-policy causal feedback search added 32 episodes but selected zero coefficients—the constant baseline.','2. [Per-tensor relative-step grid](per_tensor/README.md): 16 target-fraction pairs, 32 episodes, no numerical failures. The best fixed rule reaches 4/4 and 6/9 modes.','3. [RMS-memory and decay follow-up](rms_decay/README.md): 8 conditions, 16 episodes. Current RMS plus delayed cosine sustains ring4 from 950 (confirmed 1150); grid9 falls to 3/9 modes. RMS beta .99 gives more balanced final coverage but no sustained task.','',
'Read-only layer diagnostics exactly reproduced all metrics. At initialization the common-LR SGD generator updates its first weight matrix by 0.0095% of its RMS, output weights by 1.24%, and output bias by 25.8%. A common scalar multiplier preserves this disparity for a given gradient; this motivated per-tensor rates. It is a hypothesis about the collapse, not proof of its cause.','',
'Five analytical tests verify raw-gradient updates, absent momentum/optimizer state, unequal tensor LRs, and causal RMS memory. Source hashes match every archived phase. JSON archives use deterministic gzip (`mtime=0`); each phase includes a manifest with original-byte SHA256s and an exact source bundle.','',
'Implementation and reproduction: [SGD study](../../../benchmarks/learned_lr/sgd_study.py), [per-tensor grid](../../../benchmarks/learned_lr/relative_sgd_study.py), [RMS/decay follow-up](../../../benchmarks/learned_lr/relative_sgd_followup.py), [method notes](../../../benchmarks/learned_lr/SGD_README.md).','']
(dest/'README.md').write_text('\n'.join(lines))
shutil.copy2('/tmp/archive_pr36_sgd.py',dest/'archive_script.py')
print(dest)
