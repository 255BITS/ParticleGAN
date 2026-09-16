"""Continuous circle diagnostics from completed saved paths; never a training loss."""
import numpy as np


DEFINITION = (
    'Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*'
    '(1+(per-step signed angular error/.03)^2)), in [0,1]. '
    'Good steps require abs radial error<.1 and abs signed angular error<.03. '
    'Include the handoff transition from the last clean reference point. '
    'Good-arc turns are the longest consecutive good angular arc / (2*pi). '
    'Quality is a diagnostic, not a success probability, and does not measure absolute phase. '
    'Cold uses a single circle fit and signed speed from the first32 generated points, '
    'then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. '
    'Warm uses the true reference orbit. Report early and late scores separately.'
)


def fit_circle(paths):
    centers, radii = [], []
    for path in np.asarray(paths, dtype=np.float64):
        origin = path.mean(0)
        xy = path-origin
        fit = np.linalg.lstsq(np.column_stack((2*xy, np.ones(len(xy)))),
                             (xy*xy).sum(1), rcond=None)[0]
        centers.append(origin+fit[:2])
        radii.append(np.sqrt(max(1e-12, fit[2]+np.square(fit[:2]).sum())))
    return np.asarray(centers), np.asarray(radii)


def angles(a, b):
    return np.arctan2(a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0], (a*b).sum(-1))


def orbit_progress(generated, clean=None, prefix=None):
    x = np.asarray(generated, dtype=np.float64)
    if clean is not None:
        reference = np.asarray(clean, dtype=np.float64)
        center, radius = fit_circle(reference[:, :32])
        ref = reference-center[:, None]
        omega = angles(ref[:, 0], ref[:, 1])
        previous = reference[:, prefix-1:prefix]
        size_factor = np.ones(len(x))
    else:
        center, radius = fit_circle(x[:, :32])
        initial = x[:, :32]-center[:, None]
        initial_speed = angles(initial[:, :-1], initial[:, 1:]).mean(1)
        sign = np.where(initial_speed < 0, -1., 1.)
        omega = sign*np.clip(np.abs(initial_speed), .08, .45)
        radius_violation = np.maximum(.5-radius, 0)+np.maximum(radius-1.6, 0)
        size_factor = 1/(1+(radius_violation/.1)**2)
        previous, x = x[:, :1], x[:, 1:]
    offsets = x-center[:, None]
    preceding = np.concatenate((previous-center[:, None], offsets[:, :-1]), 1)
    angular = angles(preceding, offsets)
    radial = np.linalg.norm(offsets, axis=-1)/radius[:, None]-1
    speed_error = angular-omega[:, None]
    quality = size_factor[:, None]/((1+(radial/.1)**2)*(1+(speed_error/.03)**2))
    good = (np.abs(radial) < .1) & (np.abs(speed_error) < .03)
    if clean is None:
        good &= ((radius >= .5) & (radius <= 1.6))[:, None]
    run = np.zeros(len(x))
    longest = np.zeros(len(x))
    for valid, step in zip(good.T, angular.T):
        run = np.where(valid, run+np.abs(step), 0.)
        longest = np.maximum(longest, run)
    turns = longest/(2*np.pi)
    q = quality.mean(1)
    return {
        'quality': float(q.mean()),
        'quality_median': float(np.median(q)),
        'quality_first32': float(quality[:, :32].mean()),
        'quality_last256': float(quality[:, -256:].mean()),
        'good_step_fraction': float(good.mean()),
        'good_step_fraction_last256': float(good[:, -256:].mean()),
        'initial_good_steps_mean': float(np.cumprod(good, axis=1).sum(1).mean()),
        'longest_good_arc_turns_mean': float(turns.mean()),
        'longest_good_arc_turns_median': float(np.median(turns)),
        'quarter_turn_fraction': float((turns >= .25).mean()),
        'full_turn_fraction': float((turns >= 1).mean()),
        'per_particle_quality': q.tolist(),
        'per_particle_longest_good_arc_turns': turns.tolist(),
    }


def panels(arrays):
    result = {'cold': orbit_progress(arrays['generated'])}
    for prefix in (8, 32):
        result[f'prefix{prefix}'] = orbit_progress(arrays[f'prefix{prefix}'],
                                                  arrays['continuation_reference'], prefix)
    return result
