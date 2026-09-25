"""Opt-in CIFAR profiling; toy defaults are untouched."""
from contextlib import contextmanager
import json
import torch


class SpeedProfiler:
    """A short CPU/CUDA trace plus phase timings; disabled outside its window."""
    def __init__(self, out, start, steps):
        self.out, self.start, self.steps = out, start, steps
        self.active, self.trace, self.events = False, None, []

    def begin(self, step):
        if self.steps and step == self.start + 1:
            self.trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
            self.trace.__enter__()
            self.active = True

    @contextmanager
    def region(self, name):
        if not self.active:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.profiler.record_function(name):
            start.record()
            yield
            end.record()
        self.events.append((name, start, end))

    def end(self, step):
        if self.active and step == self.start + self.steps:
            self.trace.__exit__(None, None, None)
            torch.cuda.synchronize()
            totals = {}
            for name, start, end in self.events:
                totals[name] = totals.get(name, 0.) + start.elapsed_time(end) / self.steps
            (self.out / 'profile_phases.json').write_text(json.dumps(totals, indent=2) + '\n')
            (self.out / 'profile_ops.txt').write_text(self.trace.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
            self.trace.export_chrome_trace(str(self.out / 'trace.json'))
            print(f'PROFILE phase_ms={json.dumps(totals)}', flush=True)
            self.active, self.trace = False, None
            self.events.clear()
