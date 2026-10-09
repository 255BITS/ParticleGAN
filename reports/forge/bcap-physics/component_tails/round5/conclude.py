"""Conclude exactly the two frozen studies through Forge's public CLI."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/component_tails/queue')


def main():
    results=json.loads((OUT/'results.json').read_text())
    arms=results['arms']
    def task(arm,name):return next(r for r in arms[arm]['tasks'] if r['task_id']==name)
    width={a:task(a,'vector_unequal_width') for a in arms}
    rare={a:task(a,'vector_unequal_mass') for a in arms}
    broad={a:task(a,'vector_two_broad') for a in arms}
    comparison=(f"Matched complete outcomes: local-v2 {arms['control']['outcomes']}; prior-only {arms['candidate']['outcomes']}. "
        f"Full final width covariance {width['control']['final']['component_covariance_error']:.9f} -> "
        f"{width['candidate']['final']['component_covariance_error']:.9f}, suffix "
        f"{width['control']['terminal_passing_suffix']} -> {width['candidate']['terminal_passing_suffix']}. "
        f"Unequal-mass {rare['control']['status']} -> {rare['candidate']['status']}; "
        f"broad {broad['control']['status']} -> {broad['candidate']['status']}. "
        "Every completed job uses identical source/runtime, seed0, task laws, initial models and consumed streams; no archived score substitutes for a control.")
    conclusion=("Prior-only transport improves the width endpoint surrogate but fails full temporal width qualification and loses retained unequal-mass PASS. "
        "Gaussian endpoint KS improves while full retention/reacquisition fails. Stop this exact routing replacement; a scalar forecast observed is not a complete repair.")
    next_action=("Retain exact local-v2 scoped rare/broad evidence and original limits. Preserve center-versus-kernel decomposition and full native/temporal failures. "
        "Stop this bounded revision: no tuning, seed repeat, extension, second candidate, ordinary qualification or default promotion.")
    os.environ['PARTICLEGAN_FORGE_QUEUE']=str(QUEUE)
    for arm,study in [('control','component_tails_control_round5'),('candidate','component_tails_candidate_round5')]:
        args=[sys.executable,'-m','experiments.forge','--queue-root',str(QUEUE),'readout',arms[arm]['candidate_id'],
            '--study',study,'--conclusion',conclusion if arm=='candidate' else
            'Matched exact local-v2 primary control completed. Retain its scoped positive gates and unchanged full failures; no ordinary qualification is added.',
            '--comparison',comparison,'--next-action',next_action]
        subprocess.run(args,cwd=ROOT,check=True)


if __name__=='__main__':main()
