"""Goal-media layout controls use retained/synthetic observations, not training."""
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from benchmarks.toy_audit import api_reframe, api_run


def observations(panels):
    angle = np.linspace(0, 2*np.pi, 64)
    ring = np.column_stack((np.cos(angle), np.sin(angle)))
    long_caption = ("Public served latent law retains its enabled feature-cell perturbation. "
                    "The evaluation uses independent random streams and reports all declared "
                    "conditional modes. Output noise is a separate sampling factor. ") * 3
    scatter = dict(kind="scatter", title="Circular target geometry with a long sampling explanation",
                   target=ring, samples=ring*.9, xlabel="x", ylabel="y", caption=long_caption)
    mass = dict(kind="bar", title="All mode masses and their full denominators", xlabel="mode index",
                ylabel="probability", target=np.full(8,.125), samples=np.full(8,.125), caption=long_caption)
    views = [scatter] if panels==1 else [scatter, deepcopy(scatter), mass, deepcopy(scatter)]
    return [dict(step=step, passed=step==2, failed_bounds=[] if step==2 else ["width"],
                 metrics={"relative_distribution_error_against_full_target": .12,
                          "maximum_conditional_covariance_eigenvalue_ratio": 1.15,
                          "minimum_heldout_quality_fraction_across_panels": .90},
                 views=deepcopy(views)) for step in (0,1,2)]


@pytest.mark.parametrize("panels",(1,4))
def test_long_text_slots_do_not_overlap_or_escape_and_geometry_stays_circular(tmp_path,monkeypatch,panels):
    records=observations(panels)
    before=api_reframe._observation_identity(records)
    inspected=[]
    actual_close=plt.close
    def close(fig=None):
        if isinstance(fig,Figure):
            renderer=fig.canvas.get_renderer()
            text=[t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
            text.extend(label for ax in fig.axes for label in (ax.xaxis.label,ax.yaxis.label)
                        if label.get_text().strip())
            boxes=[t.get_window_extent(renderer) for t in text]
            for box in boxes:
                assert box.x0>=0 and box.y0>=0
                assert box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
            for index,box in enumerate(boxes):
                for other in boxes[index+1:]:
                    assert not box.overlaps(other), "caption, labels, goal or verdict overlap"
            for ax in fig.axes:
                if ax.get_xlabel()=="x":
                    points=ax.transData.transform([[0,0],[1,0],[0,1]])
                    assert np.linalg.norm(points[1]-points[0])==pytest.approx(np.linalg.norm(points[2]-points[0]))
            badge=next(t for t in text if t.get_text()=="Default test FAIL")
            assert badge.get_fontweight()=="bold"
            pixels=np.asarray(fig.canvas.buffer_rgba())
            inspected.append((pixels.shape[1],pixels.shape[0]))
        actual_close(fig)
    monkeypatch.setattr(plt,"close",close)
    path=tmp_path/"layout.gif"
    case=dict(id="long-public-sampling-layout-control",default_steps=2,
              goal="Compare true circular target geometry and full mode masses under the original sampling law, while preserving a failed default verdict even when the last observation passes.")
    result=api_run.render_gif(case,records,path,full_budget=True,requested_steps=2,final_verdict="FAIL")
    assert len(inspected)==3 and len(set(inspected))==1
    assert api_reframe._observation_identity(records)==before
    assert result["default_verdict_displayed"]=="FAIL" and result["numeric_observations_changed"] is False
    with Image.open(path) as gif:
        assert gif.n_frames==3
        assert gif.size==inspected[0]


def test_equal_aspect_excludes_scalar_samples_and_time_action_series():
    spatial=np.zeros((3,8,2))
    assert api_run._physical_geometry(dict(kind="line",target=spatial))
    assert api_run._physical_geometry(dict(kind="scatter",target=np.zeros((16,2)),xlabel="x",ylabel="y"))
    assert not api_run._physical_geometry(dict(kind="scatter",target=np.zeros((16,1))))
    assert not api_run._physical_geometry(dict(kind="line",target=np.zeros((16,2))))
    assert not api_run._physical_geometry(dict(kind="line",target=spatial,xlabel="time",ylabel="action"))
    assert not api_run._physical_geometry(dict(kind="line",target=spatial,aspect="auto"))


@pytest.mark.parametrize("shape",((8,2,1,32),(8,1,2,32)))
def test_legacy_and_current_critic_features_stack_losslessly_by_context(shape):
    values=np.arange(512,dtype=np.float32).reshape(shape)
    record=dict(step=800,passed=True,metrics={"rmse":.08},failed_bounds=[],
                views=[dict(kind="image",title="Paired critic features",target=values,samples=values+1)])
    original=api_reframe._observation_identity([record])
    annotations=[]
    prepared=api_run._display_views({"id":"api-critic-lag-current"},[record],{},annotations)[0][0]
    for name in ("target","samples"):
        assert prepared[name].shape==(1,1,16,32)
        np.testing.assert_array_equal(prepared[name].reshape(shape),record["views"][0][name])
        assert api_run._image_grid(prepared[name]).shape==(16,32)
    assert "two adjacent grayscale feature rows; first 8 contexts" in prepared["caption"]
    assert annotations[0]["lossless_feature_layout"]["values_modified"] is False
    assert api_reframe._observation_identity([record])==original
