from types import SimpleNamespace
import torch
from experiments.memory_handoff_scout import Config
from experiments.memory_recent import RecentWriter, LocalReader


def test_recent_slots_preserve_order_and_writer_gradients():
    writer = RecentWriter(32, 4)
    points = torch.randn(2, 6, 2, requires_grad=True)
    memory = writer.initial(points)
    for point in points.unbind(1):
        memory = writer.write(memory, point)
    torch.testing.assert_close(memory.flatten(1)[:, 24:], points[:, -4:].flip(1).flatten(1))
    memory.sum().backward()
    assert points.grad[:, :2].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in writer.parameters())


def test_residual_reader_uses_only_latest_point_as_base_and_bounds_increment():
    cfg = SimpleNamespace(**{**vars(Config()), 'recent_points':4, 'residual_output':True, 'output_bound':3.})
    reader = LocalReader(cfg)
    memory = torch.randn(3, 8, 4)
    latest = memory.flatten(1)[:, 24:26]
    with torch.no_grad():
        for p in reader.parameters():
            p.zero_()
    point, _ = reader(torch.randn(3, 4), memory)
    torch.testing.assert_close(point, latest)
    with torch.no_grad():
        reader.net[-1].bias.fill_(100.)
    point, _ = reader(torch.randn(3, 4), memory)
    torch.testing.assert_close(point, latest+3)
