"""Run a K3P driver with deterministic ortho init (K3P_ORTHO=1) and a seed offset (K3P_SEED_OFFSET).
K3P_INIT_ONLY=1: stop at the first optimizer step (after all optimizers built) and only write the init log."""
import atexit, os, runpy, sys
OFF = int(os.environ.get('K3P_SEED_OFFSET', '0'))
script = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
import torch
if OFF:
    shift = lambda s: (int(s) + OFF) % (2 ** 63)
    _cuda_ms, _cuda_msa = torch.cuda.manual_seed, torch.cuda.manual_seed_all
    def manual_seed(seed):
        s = shift(seed)
        if not torch.cuda._is_in_bad_fork():
            _cuda_msa(s)
        return torch.default_generator.manual_seed(s)
    torch.manual_seed = torch.random.manual_seed = manual_seed
    torch.cuda.manual_seed = lambda seed: _cuda_ms(shift(seed))
    torch.cuda.manual_seed_all = lambda seed: _cuda_msa(shift(seed))
    _Base = torch.Generator
    class Generator(_Base):
        def manual_seed(self, seed):
            return super().manual_seed(shift(seed))
    torch.Generator = Generator
if os.environ.get('K3P_ORTHO') == '1':
    out = sys.argv[sys.argv.index('--output') + 1]
    repo = sys.argv[sys.argv.index('--repo') + 1] if '--repo' in sys.argv else os.getcwd()
    sys.path.insert(0, repo); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ortho_init
    ortho_init.install()
    def _dump():
        os.makedirs(out, exist_ok=True); ortho_init.dump(os.path.join(out, 'ortho-init.json'))
    atexit.register(_dump)
    if os.environ.get('K3P_INIT_ONLY') == '1':
        orig_step = torch.optim.Adam.step
        def step(self, *a, **k):
            _dump(); os._exit(0)
        torch.optim.Adam.step = step
runpy.run_path(script, run_name='__main__')
