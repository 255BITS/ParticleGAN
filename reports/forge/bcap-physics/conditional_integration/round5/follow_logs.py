"""Follow dynamically created worker stdout without touching queue state."""
from pathlib import Path
import json,time
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/conditional_integration/queue')
PROGRESS=Path('/tmp/bcap-physics-round5-20261009/conditional_integration/progress.json')
LOG=QUEUE.parent/'logs/driver.log'

def main():
    offsets={}
    while True:
        for path in sorted(QUEUE.glob('*/*/run.log')):
            with path.open() as stream:
                stream.seek(offsets.get(path,0));new=stream.read();offsets[path]=stream.tell()
            if new:
                with LOG.open('a') as output:
                    output.write(f'\nworker={path.parent.name}\n{new}');output.flush()
        if json.loads(PROGRESS.read_text())['phase'] in ('trained','published','complete'):break
        time.sleep(2)
if __name__=='__main__':main()
