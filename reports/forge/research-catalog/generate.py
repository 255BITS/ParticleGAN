"""Render a frozen evidence catalog. Standard-library metadata only; no science."""
from pathlib import Path
import argparse,csv,io,json
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parent

def read(name):return json.loads((ROOT/name).read_text())
def text(value,limit=None):
    if isinstance(value,(dict,list)):value=json.dumps(value,ensure_ascii=False,sort_keys=True)
    value=str(value if value is not None else 'UNRECORDED').replace('\n',' ').replace('|','\\|').replace('<','&lt;').replace('>','&gt;')
    if limit and len(value)>limit:value=value[:limit-1]+'…'
    return value

def validate(attempts,studies,coverage,pins):
    assert len(attempts)==coverage['verified_recent_attempts']==285
    assert len({r['attempt_id'] for r in attempts})==285
    assert dict(Counter(r['status'] for r in attempts))==coverage['status_counts']
    assert {int(k):v for k,v in coverage['publication_partition'].items()}==dict(Counter(r['publication']['pr'] for r in attempts))
    assert len(studies)==coverage['older_study_records']==312
    assert len({s['study_id'] for s in studies})==312
    assert dict(Counter(s['record_kind'] for s in studies))==coverage['older_kinds']
    assert coverage['coverage_complete'] is False and coverage['ordinary_qualification'] is False
    operational=read('operational-records.json')
    assert len(operational)==17 and all(r['scientific_utility_credited'] is False for r in operational)
    assert dict(Counter(r['record_kind'] for r in operational))=={
        'invalid_pre_runtime_execution':12,'cpu_preflight_rejection':3,'bounded_mechanism_prefix':2}
    ids={r['attempt_id'] for r in attempts}
    for r in attempts:
        assert r['status'] in {'PASS','FAIL','INCOMPLETE'}
        assert len(r['source_digest'])==64 and len(r['candidate_revision'])==64
        assert r['publication']['status'] in {'MERGED_DEVELOP','PUBLISHED_DRAFT'}
        assert r['publication']['url'].startswith('https://github.com/255BITS/ParticleGAN/blob/')
        assert r['publication']['conditions_sha256'] and r['publication']['recipe_sha256']
        assert all(p in pins for p in r['archive'].values())
        assert all(b['attempt_id'] in ids for b in r['declared_baseline_attempts'])
        if r['status']=='INCOMPLETE':assert r['helpful_lesson'] is None and r['operational_limitation']
    for s in studies:
        if s['record_kind']=='unrun':assert 'no scientific utility' in s['utility_scope']
    return {'canonical_attempts':285,'older_studies':312,'duplicate_attempt_ids':0,
            'incomplete_scientific_utility_claims':0,'qualification_reuse':False}

def render():
    attempts,studies,coverage,pins=read('attempts.json'),read('studies.json'),read('coverage.json'),read('archive-pins.json')
    result=validate(attempts,studies,coverage,pins)
    groups=defaultdict(list)
    for r in attempts:groups[r['cohort']].append(r)
    order=sorted(groups,key=lambda k:min(r['terminal_utc'] or '9999' for r in groups[k]))
    lines=['# Canonical recent attempt catalog','',
        '[Readable entry point](README.md) · [Complete metadata](attempts.json) · [CSV](run-register.csv)','',
        '285 distinct IDs from the frozen recent register. PASS/FAIL is the original complete verdict. '
        'INCOMPLETE receives operational lessons only. Cohort lessons are shared conclusions, not 285 independent innovations. '
        'The linked original receipts retain complete recipe, task, source, seed, prior, serving, gates and costs; summaries never replace them.','']
    for name in order:
        rs=groups[name];first=rs[0];c=Counter(r['status'] for r in rs)
        lines+=['## '+name,'',f'{len(rs)} attempts; '+', '.join(f'{k} {c[k]}' for k in ['PASS','FAIL','INCOMPLETE'] if c[k])+'.','',
            '**Recorded lesson:** '+text(first['helpful_lesson'] or first['operational_limitation']),
            '', '**Next implication:** '+text(first['next_implication']),'',
            '| Canonical attempt / task / recipe | Declared hypothesis and change | Declared baseline | Original result | Publication / full receipt |',
            '|---|---|---|---|---|']
        for r in rs:
            baselines=r['declared_baseline_attempts']
            baseline='; '.join('`'+b['attempt_id']+'` '+b['status']+' ('+b['relation']+')' for b in baselines) or text(r['declared_control'])+'; matching not reconstructed'
            lines.append('| `'+r['attempt_id']+'`<br>'+text(r['task'])+'<br>'+text(r['candidate'])+
                ' | '+text(r['declared_hypothesis'],210)+'<br>'+text(r['declared_change'],210)+
                ' | '+baseline+' | **'+r['status']+'**; new updates '+text(r['new_updates'])+
                ' | [PR'+str(r['publication']['pr'])+' · '+r['publication']['status']+']('+r['publication']['url']+') |')
        lines+=['','Exact source/revision/protocol hashes and archive locators for every row are in [attempts.json](attempts.json).','']
    attempts_md='\n'.join(lines)+'\n'

    lines=['# Older source-bound study catalog','',
        '[Entry point](README.md) · [Full original comparison metadata](studies.json) · [65 original arms](original-arms.json)','',
        '312 archived study identities, not 312 independent training runs. Shared controls, repeats, regrades and analyses '
        'overlap. No canonical execution UUID has been invented for the older records. The source ledger supplies '
        '100 trained comparisons, 77 diagnostics, 129 full-horizon diagnostics and 6 unrun records. '
        'Full-horizon diagnostics retain their original limitations and do not acquire ordinary qualification.','',
        '| Study / original kind | Hypothesis / change | Recorded result and lesson | Next implication / evidence |',
        '|---|---|---|---|']
    for s in studies:
        results=[]
        for c in s['comparisons']:
            b,a=c['baseline'],c['candidate']
            results.append(text(c['task'])+': '+text(b.get('status'))+'→'+text(a.get('status'))+
                '; matched='+text(c.get('matched_contract_verified')))
        original=s['original_report']
        pin_info=pins.get(original,{}) if original else {}
        evidence=('`'+original+'`; '+pin_info.get('availability','UNAVAILABLE')+'; sha256 `'+str(pin_info.get('sha256'))+'`') if original else 'Original human report unavailable; retained ledger metadata remains'
        lines.append('| `'+s['study_id']+'`<br>'+text(s['record_kind'])+'<br>'+text(s['title'],150)+
            ' | '+text(s['declared_hypothesis'],180)+'<br>'+text(s['declared_change'],180)+
            ' | '+text('; '.join(results),500)+'<br>'+text(s['recorded_decision'],380)+'<br>'+text(s['utility_scope'])+
            ' | '+text(s['next_implication'],260)+'<br>'+evidence+' |')
    studies_md='\n'.join(lines)+'\n'

    stream=io.StringIO();fields=['attempt_id','cohort','candidate','candidate_revision','source_digest','task','terminal_utc',
        'status','new_updates','declared_hypothesis','declared_change','declared_control','declared_baseline_attempts',
        'helpful_lesson','operational_limitation','next_implication','publication','archive','protocol_summary']
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for r in attempts:writer.writerow({k:json.dumps(r[k],sort_keys=True) if isinstance(r[k],(list,dict)) else r[k] for k in fields})
    return {'ATTEMPTS.md':attempts_md,'STUDIES.md':studies_md,'run-register.csv':stream.getvalue()},result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    outputs,result=render();stale=[]
    for name,t in outputs.items():
        p=ROOT/name
        if not p.exists() or p.read_text()!=t:
            stale.append(name)
            if not args.check:p.write_text(t)
    if args.check and stale:parser.exit(1,'Stale catalog: '+', '.join(stale)+'\n')
    print(json.dumps({**result,'changed':stale,'training_launched':False,'sampling_launched':False}))

if __name__=='__main__':main()
