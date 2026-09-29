import sys, math, torch
sys.argv = ['x', '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17']
src = open('/ml2/hypergan/gan-attempts/noout-20260928/design/review_E17_code/test_E17_extra.py').read()
head = src[:src.index('# ============================================================== A.')]
tail_start = src.index('# ============================================================== C.')
ring = src[tail_start: src.index('# C1 the flagged set')]
exec(head); exec(ring)
t, bd, real, mode = ring_table(0)
print('after ring_table(0): iso counters', {k: v for k, v in bd.counters.items() if k.startswith('iso')}, 'rows_since_eval', bd.rows_since_eval)
for i in range(70):
    t.step(real(6000 + i))
print('iso_log', bd.iso_log)
print('counters', {k: v for k, v in bd.counters.items() if k.startswith('iso') or k in ('evals','moves','stale_resets')})
