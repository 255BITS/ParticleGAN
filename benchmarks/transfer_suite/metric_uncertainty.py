"""Gate-metric uncertainty: oracle noise floors and the evaluation-draw contracts.

Reporting only. Nothing here changes a gate decision or an existing metric value,
and the frozen evaluator sources (``vector_tasks.py`` and ``image_tasks.py``, whose
SHA-256 digests Forge task cards pin) are not modified.

Oracle noise floor (``oracle-noise-floor-v1``)
    A gate metric measured on a finite evaluation draw is noisy even for a perfect
    sampler. The floor scores ``ORACLE_DRAWS`` independent draws from the target law
    itself (the oracle) with the task's own scorer, against the same reference draw
    and projections the candidate was scored with, and reports for every gate:

    * ``oracle_mean`` / ``oracle_std``: the metric's finite-n value and spread for a
      perfect sampler. For distances such as sliced W1 the mean is the finite-n bias;
    * ``oracle_pass_fraction``: how often a perfect sampler passes this gate;
    * ``margin_oracle_std``: the candidate's signed distance to the threshold in
      oracle standard deviations (positive is the passing side);
    * ``oracle_margin_oracle_std``: the same distance for the oracle mean.

    Oracle draws use their own generators, seeded from a hash of the floor version
    and the reference seed, so they never consume training, data, latent, eval-stream
    or global RNG state. Gates the oracle scorer does not produce (for example a
    wrapper's extra metric) are listed in ``unscored_gates``, never given a floor.

Evaluation-draw contracts (vector tasks)
    ``fixed_v1`` is the archived law and the default: every observation reuses the
    reference target draw (seed 991), the sliced-W1 projections (seed 992) and, in
    ``vector_tasks.run_episode``, the eval latents (seed 990). Terminal observations
    of one episode therefore differ only through the model and are strongly
    correlated. ``per_observation_v2`` is an explicit, versioned opt-in: those
    streams are re-derived from each observation's completed step, so successive
    observations are closer to independent. Switching it on starts a new evaluation
    cohort; its receipts are not comparable with archived ``fixed_v1`` receipts.
"""
import hashlib
import json
import math

import torch

from benchmarks.locked_shared.observation import threshold_margin

NOISE_FLOOR_VERSION = "oracle-noise-floor-v1"
ORACLE_DRAWS = 16
EVAL_DRAW_CONTRACTS = ("fixed_v1", "per_observation_v2")
LEGACY_EVAL_SEEDS = dict(latent=990, reference=991, projection=992)
# Projection count of the frozen vector scorer (sliced_w1(fake, real, 32, seed=992)).
SW1_PROJECTIONS = 32
# Ungated metrics that still receive an oracle mean/std next to the gates.
VECTOR_REPORT_METRICS = ("sw1_normalized", "mean_error", "covariance_error")
IMAGE_REPORT_METRICS = ("mean_rmse", "distribution_tv")
_CACHE_LIMIT = 256
_VECTOR_ORACLE = {}
_IMAGE_ORACLE = {}


def derived_seed(*parts) -> int:
    """Stable 60-bit seed for an isolated evaluation stream named by ``parts``."""
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:15], 16)


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _ratio(numerator, denominator):
    if not _number(numerator) or not _number(denominator) or denominator <= 0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _scalars(metrics):
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def summarize(metrics, oracle_rows, gates, *, report=()):
    """Annotate ``metrics`` gates with the oracle floor estimated from ``oracle_rows``.

    ``gates`` are ``(key, op, bound)`` triples; ``report`` lists extra keys that get
    only the oracle mean/std. ``oracle_all_gates_pass_fraction`` is over the gates
    the oracle scorer produced.
    """
    entries, unscored = {}, []
    keys = [(key, op, bound) for key, op, bound in gates]
    keys += [(key, None, None) for key in report if key not in {k for k, _, _ in keys}]
    for key, op, bound in keys:
        values = [float(row[key]) for row in oracle_rows if _number(row.get(key))]
        if not values:
            if op is not None:
                unscored.append(key)
            continue
        mean = sum(values) / len(values)
        std = (math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
               if len(values) > 1 else None)
        entry = dict(oracle_mean=mean, oracle_std=std, oracle_draws=len(values))
        if op is not None:
            value = metrics.get(key)
            entry.update(op=op, threshold=bound,
                         oracle_pass_fraction=sum(threshold_margin(x, op, bound) >= 0 for x in values) / len(values),
                         margin_oracle_std=_ratio(threshold_margin(value, op, bound), std) if _number(value) else None,
                         oracle_margin_oracle_std=_ratio(threshold_margin(mean, op, bound), std))
        entries[key] = entry
    scored = [(k, op, b) for k, op, b in gates if k not in unscored]
    joint = [all(_number(row.get(k)) and threshold_margin(row[k], op, b) >= 0 for k, op, b in scored)
             for row in oracle_rows]
    return dict(version=NOISE_FLOOR_VERSION, draws=len(oracle_rows), metrics=entries,
                unscored_gates=unscored,
                oracle_all_gates_pass_fraction=sum(joint) / len(joint) if joint else None)


def compact(metrics):
    """``metrics`` with any nested ``noise_floor`` reduced to a tail-friendly summary.

    For log lines (Forge observation events, suite DONE lines): keeps the floor
    version, ``oracle_all_gates_pass_fraction`` and each gate's ``margin_oracle_std``.
    The full floor stays in the stored evidence. Non-dict input is returned as is.
    """
    if not isinstance(metrics, dict):
        return metrics
    out = {}
    for key, value in metrics.items():
        if key == "noise_floor" and isinstance(value, dict):
            gates = {name: entry.get("margin_oracle_std") for name, entry in (value.get("metrics") or {}).items()
                     if isinstance(entry, dict) and "op" in entry}
            out[key] = dict(version=value.get("version"),
                            oracle_all_gates_pass_fraction=value.get("oracle_all_gates_pass_fraction"),
                            margin_oracle_std=gates)
        else:
            out[key] = compact(value) if isinstance(value, dict) else value
    return out


def _remember(cache, key, build):
    if key not in cache:
        if len(cache) >= _CACHE_LIMIT:
            cache.clear()
        cache[key] = build()
    return [dict(row) for row in cache[key]]


# ----------------------------------------------------------------------------- vectors

def evaluation_seeds(contract, completed_steps):
    """Seeds of the isolated eval latent, reference-draw and projection streams."""
    if contract == "fixed_v1":
        return dict(LEGACY_EVAL_SEEDS)
    if contract == "per_observation_v2":
        return {name: derived_seed(contract, name, seed, int(completed_steps))
                for name, seed in LEGACY_EVAL_SEEDS.items()}
    raise ValueError(f"unsupported eval_draw_contract: {contract!r}; expected one of {EVAL_DRAW_CONTRACTS}")


@torch.no_grad()
def score_against_reference(fake, real, spec, completed_steps, projection_seed):
    """``vector_tasks.score_samples`` metrics against an explicit reference draw.

    A transcription of the frozen scorer body with the reference draw and projection
    seed made explicit (the frozen file stays byte-identical because Forge task cards
    pin it). Tests bind it to the frozen scorer: at reference seed 991 and projection
    seed 992 both return identical dictionaries for every predeclared task.
    """
    from lib.toy_metrics import sliced_w1
    from .vector_tasks import CORE_MAHALANOBIS2, SPILL_MAHALANOBIS2, component_resolved, target_scale
    centered = real-real.mean(0)
    scale = centered.square().sum(1).mean().sqrt().clamp_min(1e-8)
    rcov = centered.T@centered/len(real)
    fcenter = fake-fake.mean(0)
    fcov = fcenter.T@fcenter/len(fake)
    result = dict(sw1_normalized=sliced_w1(fake, real, SW1_PROJECTIONS, seed=projection_seed)/float(scale),
                  mean_error=float((fake.mean(0)-real.mean(0)).norm()/scale),
                  covariance_error=float((fcov-rcov).norm()/rcov.norm().clamp_min(1e-8)),
                  target_scale=float(scale), sample_count=len(fake))
    if spec["kind"] == "gaussian_mixture" and spec["identifiable"]:
        units = target_scale(spec, completed_steps)
        means = torch.tensor(spec["means"])*units
        cov = torch.tensor(spec["covariances"])*units**2
        target = torch.tensor(spec["masses"])
        assignment = torch.cdist(fake, means).argmin(1)
        counts = torch.bincount(assignment, minlength=len(means))
        mass = counts/len(fake)
        delta = fake-means[assignment]
        mahal = torch.einsum("ni,nij,nj->n", delta, torch.linalg.inv(cov)[assignment], delta)
        def shape(points, k):
            if len(points) < 10:
                return 1., 0.
            x = points-points.mean(0)
            empirical = x.T@x/len(points)
            inverse = torch.linalg.inv(torch.linalg.cholesky(cov[k]))
            return (float((empirical-cov[k]).norm()/cov[k].norm()),
                    float(torch.linalg.eigvalsh(inverse@empirical@inverse.T).min()))

        errors, eigen_ratios, core_errors, core_ratios, spills = [], [], [], [], []
        for k in range(len(means)):
            member = assignment == k
            error, ratio = shape(fake[member], k)
            errors.append(error)
            eigen_ratios.append(ratio)
            error, ratio = shape(fake[member & (mahal <= CORE_MAHALANOBIS2)], k)
            core_errors.append(error)
            core_ratios.append(ratio)
            spills.append(float((mahal[member] > SPILL_MAHALANOBIS2).float().mean()) if member.sum() >= 10 else 1.)
        result.update(mass_tv=float((mass-target).abs().sum()/2),
                      min_mass_ratio=float((mass/target).min()), hq=float((mahal <= 9).float().mean()),
                      component_covariance_error=sum(errors)/len(errors), component_covariance_errors=errors,
                      component_min_eigen_ratio=min(eigen_ratios),
                      component_core_covariance_error=sum(core_errors)/len(core_errors),
                      component_core_covariance_errors=core_errors,
                      component_core_min_eigen_ratio=min(core_ratios), component_core_eigen_ratios=core_ratios,
                      max_component_spill=max(spills), component_spill=spills,
                      component_mass=mass.tolist(), target_mass=target.tolist(), component_counts=counts.tolist())
        resolved = component_resolved(spec)
        kept = [k for k, ok in enumerate(resolved) if ok]
        result.update(component_resolved=resolved,
                      resolved_core_covariance_error=sum(core_errors[k] for k in kept)/len(kept) if kept else 0.,
                      resolved_core_min_eigen_ratio=min((core_ratios[k] for k in kept), default=1.),
                      resolved_max_component_spill=max((spills[k] for k in kept), default=0.))
    return result


def _reference(spec, count, completed_steps, seeds):
    from .vector_tasks import sample_target
    return sample_target(spec, count, torch.Generator().manual_seed(seeds["reference"]), completed_steps)


def _vector_oracle_rows(spec, count, completed_steps, seeds):
    from .vector_tasks import sample_target, target_scale
    key = (json.dumps(spec, sort_keys=True, default=repr), int(count),
           float(target_scale(spec, completed_steps)), seeds["reference"], seeds["projection"])

    def build():
        real = _reference(spec, count, completed_steps, seeds)
        rows = []
        for draw in range(ORACLE_DRAWS):
            oracle = sample_target(spec, count, torch.Generator().manual_seed(
                derived_seed(NOISE_FLOOR_VERSION, "vector", seeds["reference"], draw)), completed_steps)
            rows.append(_scalars(score_against_reference(oracle, real, spec, completed_steps, seeds["projection"])))
        return rows
    return _remember(_VECTOR_ORACLE, key, build)


def vector_noise_floor(metrics, spec, completed_steps, count, *, contract="fixed_v1"):
    """Oracle floor of every gate in ``spec["thresholds"]`` for ``count`` samples."""
    seeds = evaluation_seeds(contract, completed_steps)
    floor = summarize(metrics, _vector_oracle_rows(spec, count, completed_steps, seeds),
                      spec["thresholds"], report=VECTOR_REPORT_METRICS)
    floor.update(kind="vector", sample_count=int(count), eval_draw_contract=contract,
                 reference_seed=seeds["reference"], projection_seed=seeds["projection"],
                 sw1_projections=SW1_PROJECTIONS,
                 oracle_seed_rule=f"derived_seed({NOISE_FLOOR_VERSION!r}, 'vector', reference_seed, draw)")
    return floor


def with_vector_noise_floor(metrics, spec, completed_steps, count, *, contract="fixed_v1"):
    """A copy of ``metrics`` with the reporting-only floor fields added.

    Adds ``noise_floor``, ``sw1_projections`` and ``sw1_finite_n_bias`` (the oracle
    mean of ``sw1_normalized``: what a perfect sampler scores at this sample count,
    reference draw and projection set). Existing keys and values are untouched.
    """
    floor = vector_noise_floor(metrics, spec, completed_steps, count, contract=contract)
    bias = floor["metrics"].get("sw1_normalized", {}).get("oracle_mean")
    return dict(metrics, sw1_projections=SW1_PROJECTIONS, sw1_finite_n_bias=bias, noise_floor=floor)


@torch.no_grad()
def score_vector(fake, spec, completed_steps, *, contract="fixed_v1", noise_floor=True):
    """Score a vector-task draw under a declared evaluation-draw contract.

    ``fixed_v1`` calls the frozen ``vector_tasks.score_samples`` unchanged.
    ``per_observation_v2`` scores against a reference draw and projections derived
    from ``completed_steps`` (a new, non-comparable evaluation cohort).
    """
    from .vector_tasks import score_samples
    seeds = evaluation_seeds(contract, completed_steps)
    if contract == "fixed_v1":
        metrics = score_samples(fake, spec, completed_steps)
    elif not torch.isfinite(fake).all():
        metrics = {key: None for key, _, _ in spec["thresholds"]}
    else:
        metrics = score_against_reference(fake, _reference(spec, len(fake), completed_steps, seeds), spec,
                                          completed_steps, seeds["projection"])
    if noise_floor:
        metrics = with_vector_noise_floor(metrics, spec, completed_steps, len(fake), contract=contract)
    return metrics


def evaluation_contract(contract="fixed_v1"):
    """Receipt block naming the evaluation draws and the floor that annotated them."""
    evaluation_seeds(contract, 0)
    return dict(eval_draw_contract=contract, noise_floor=NOISE_FLOOR_VERSION, oracle_draws=ORACLE_DRAWS,
                sw1_projections=SW1_PROJECTIONS, comparable_with_archived_fixed_v1=contract == "fixed_v1")


def annotate_vector_episode(result, spec, *, count=None):
    """Add floors to a finished ``vector_tasks.run_episode`` result in place.

    Uses each observation's own step and sample count (``sample_count`` of the row,
    falling back to ``count`` or ``vector_tasks.EVAL_SAMPLES``); no model is sampled
    again. A reporting failure is recorded, never raised into the episode.
    """
    from . import vector_tasks
    if result.get("error") or not result.get("observations"):
        return result
    try:
        cfg = vector_tasks.resolve(spec, allow_reserved=True)
        default = vector_tasks.EVAL_SAMPLES if count is None else count

        def size(metrics):
            n = metrics.get("sample_count")
            return int(n) if _number(n) and n > 0 else default
        last = result["observations"][-1]["step"]
        for row in result["observations"]:
            ema = row.get("ema")
            plain = {k: v for k, v in row.items() if k != "ema"}
            row.update(with_vector_noise_floor(plain, cfg, row["step"], size(plain)))
            if isinstance(ema, dict):
                row["ema"] = with_vector_noise_floor(ema, cfg, row["step"], size(ema))
        for name in ("live", "ema"):
            if isinstance(result.get(name), dict) and result[name]:
                result[name] = with_vector_noise_floor(result[name], cfg, last, size(result[name]))
        result["evaluation_contract"] = evaluation_contract("fixed_v1")
    except Exception as error:  # reporting only: keep the measured episode intact
        result["noise_floor_error"] = f"{type(error).__name__}: {error}"
    return result


# ------------------------------------------------------------------------------ images

@torch.no_grad()
def image_noise_floor(metrics, spec, centers, count):
    """Oracle floor of the image gates (``modes``, ``hq``) for ``count`` images.

    The oracle is ``count`` independent draws from the training law (clipped noisy
    templates), the number of images the enumerated particle table produces, scored
    by the frozen ``image_tasks.image_metrics``.
    """
    from .image_tasks import image_metrics
    from .protocol import requirements
    thresholds = spec["thresholds"]
    key = (json.dumps(spec, sort_keys=True, default=repr), int(count))

    def build():
        rows = []
        for draw in range(ORACLE_DRAWS):
            rng = torch.Generator(device=centers.device).manual_seed(
                derived_seed(NOISE_FLOOR_VERSION, "image", spec.get("name"), draw))
            index = torch.randint(len(centers), (int(count),), generator=rng, device=centers.device)
            chosen = centers[index]
            images = (chosen + spec["noise_std"] * torch.randn(
                chosen.shape, generator=rng, device=centers.device)).clamp(0., 1.)
            rows.append(_scalars(image_metrics(images, centers, thresholds)))
        return rows
    floor = summarize(metrics, _remember(_IMAGE_ORACLE, key, build), requirements(spec),
                      report=IMAGE_REPORT_METRICS)
    floor.update(kind="image", sample_count=int(count), oracle="clipped noisy templates, one draw per particle",
                 oracle_seed_rule=f"derived_seed({NOISE_FLOOR_VERSION!r}, 'image', name, draw)")
    return floor


def annotate_image_episode(result, spec):
    """Add floors to a finished ``image_tasks.run_episode`` result in place."""
    from .image_tasks import templates
    if result.get("error") or not result.get("observations"):
        return result
    try:
        centers, count = templates(spec), spec["particles"]
        for row in result["observations"]:
            ema = row.get("ema")
            row["noise_floor"] = image_noise_floor(row, spec, centers, count)
            if isinstance(ema, dict):
                row["ema"] = dict(ema, noise_floor=image_noise_floor(ema, spec, centers, count))
        for name in ("live", "ema"):
            if isinstance(result.get(name), dict) and result[name]:
                result[name] = dict(result[name], noise_floor=image_noise_floor(result[name], spec, centers, count))
        # The image evaluator has no selectable draw contract: it enumerates the
        # particle table, so its archived evaluation law is recorded as fixed_v1.
        result["evaluation_contract"] = dict(eval_draw_contract="fixed_v1", noise_floor=NOISE_FLOOR_VERSION,
                                             oracle_draws=ORACLE_DRAWS,
                                             comparable_with_archived_fixed_v1=True)
    except Exception as error:  # reporting only: keep the measured episode intact
        result["noise_floor_error"] = f"{type(error).__name__}: {error}"
    return result
