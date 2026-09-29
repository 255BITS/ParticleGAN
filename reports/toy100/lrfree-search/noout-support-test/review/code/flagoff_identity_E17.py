"""Flag-off pkg-E17 == pkg-E14 over a long run with many ordinary birth-death moves (fixture table of 200 rows, 800 steps): full state_dict bit comparison (recipe excluded)."""
import sys, os, math, torch
torch.set_num_threads(int(os.environ.get('NT', 2)))
sys.argv = ['x']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replay_E17.py')).read()
exec(src[:src.index("def build(")] .replace("FORCED = '--forced-serve' in sys.argv", "FORCED = False"))       # imports, fresh(), constants
from fixture import make
def run(pkg, gate, steps=800):
    fresh(); b, real, digest = make(pkg, birth_death_space='critic', row_evidence_gate=gate, row_evidence_null='scaled', table_release_rule='anchor', reopen_signal='none', serve_average=4.0); t = b()
    for i in range(steps): t.step(real(i))
    return t
exec(src[src.index("def diff("): src.index("allok = True")])
for gate in (True, False):
    a, b = run(E17, gate), run(E14, gate); c = a.birth_death.counters
    d = diff(a.state_dict(), b.state_dict(), skip=('recipe',))
    print(('[BIT-EXACT] ' if not d else '[DIFFER] ') + f'flag OFF pkg-E17 == pkg-E14, gate {gate}, 800 steps at N=200: evaluations {c["evals"]}, ordinary moves {c["moves"]}, stale resets {c["stale_resets"]}, last_decisive {getattr(a._table_tester(), "last_decisive", None)}' + ('' if not d else ' | ' + '; '.join(d[:3])))
