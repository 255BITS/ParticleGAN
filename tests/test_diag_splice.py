"""Scheduler rules for the ring sample-splice hook. No training."""
from benchmarks.toy100.splice_hook import GROUPS, parse_window, resolve_streams


class _Source:
    """Same rule as Controller.source, without a tape or a probe."""

    def __init__(self, selected, switch=10**9, window=None):
        self.selected = selected
        self.switch = switch
        self.window = window

    def source(self, step, stream):
        if self.selected is not None and stream not in self.selected:
            return "a"
        if self.window is not None:
            start, stop = self.window
            return "b" if start <= step < stop else "a"
        return "b" if step >= self.switch else "a"


def test_stream_groups_cover_named_draws():
    assert "data_d_idx" in GROUPS["data"] and "prior_g" in GROUPS["prior"]
    assert resolve_streams("all") is None
    assert resolve_streams("data,prior") == GROUPS["data"] | GROUPS["prior"]
    assert resolve_streams("output_train") == {"output_train"}


def test_switch_and_window_are_half_open():
    switched = _Source(None, switch=400)
    assert switched.source(399, "prior_d") == "a"
    assert switched.source(400, "prior_d") == "b"
    window = _Source(resolve_streams("data"), window=parse_window("200:250"))
    assert window.source(200, "data_d_idx") == "b"
    assert window.source(249, "data_g_eps") == "b"
    assert window.source(250, "data_d_idx") == "a"
    assert window.source(200, "prior_d") == "a"
    assert parse_window(None) is None
