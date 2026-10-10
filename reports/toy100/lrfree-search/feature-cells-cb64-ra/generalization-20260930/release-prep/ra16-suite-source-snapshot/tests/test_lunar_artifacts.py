"""Evidence exports keep simulator time and never promote a crash as speed."""
import numpy as np
import pytest

from lib.lunar_artifacts import episode_summary, render_comparison


def _episode(seed, steps, outcome, shade):
    times = [0] + list(range(3, steps + 1, 3))
    if times[-1] != steps:
        times.append(steps)
    frames = np.stack([np.full((16, 20, 3), (shade + i * 13) % 255, np.uint8)
                       for i in range(len(times))])
    power = np.zeros(steps, np.float32)
    power[1::2] = .5
    return dict(seed=seed, steps=steps, outcome=outcome, return_=float(steps),
                contact_step=steps, variant="test-bidirectional", frames=frames,
                frame_stride=3, actions=np.column_stack((np.zeros(steps), np.zeros(steps))),
                downward_main_power=power)


def test_gifs_keep_real_timing_and_terminal_residual(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    slow, fast = _episode(41, 8, "successful_landing", 20), _episode(41, 5, "successful_landing", 100)
    assets = render_comparison(tmp_path, slow, fast, {"checkpoint": "checkpoints/fast.pt"})
    assert assets == dict(slow_gif="slow.gif", fast_gif="fast.gif",
                          comparison_gif="comparison.gif", dashboard="index.html")
    totals = {}
    for name in ("slow", "fast", "comparison"):
        with Image.open(tmp_path / f"{name}.gif") as im:
            durations = []
            for index in range(im.n_frames):
                im.seek(index)
                durations.append(im.info["duration"])
            totals[name] = sum(durations)
    # 50 simulator steps/s: 8 and 5 steps, plus the same 3-step final hold.
    assert totals == {"slow": 220, "fast": 160, "comparison": 220}
    assert (tmp_path / "index.html").is_file()


def test_dashboard_escapes_report_and_rejects_mismatched_worlds(tmp_path):
    pytest.importorskip("PIL.Image")
    slow, fast = _episode(7, 4, "successful_landing", 10), _episode(7, 2, "crash", 20)
    render_comparison(tmp_path, slow, fast, {
        "checkpoint": "<script>alert(1)</script>",
        "leaderboard": [{"controller": "<bad>", "success_rate": 0.0}],
    })
    html = (tmp_path / "index.html").read_text()
    assert "&lt;script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Flight leaderboard" in html and "&lt;bad&gt;" in html
    assert "FAST LANDED" not in html
    assert episode_summary(fast)["downward_boost_steps"] == 1
    fast["seed"] = 8
    with pytest.raises(ValueError, match="same reset seed"):
        render_comparison(tmp_path, slow, fast, {})
