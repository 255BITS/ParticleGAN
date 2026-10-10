"""Raw-gradient anticipation with one historical gradient per owned coordinate.

The first application uses g; later applications use 2*g-g_previous. Prior
rows have independent histories indexed by actual sampled IDs, so unsampled
rows neither move nor acquire dense regularizer/standardization history.
This is a stochastic alternating-game hypothesis, not an extragradient proof.
"""
import torch

PREVIOUS = "optimism_previous_gradient"
SEEN = "optimism_seen_rows"
COUNTERS = ("steps", "parameter_applications", "extrapolated_applications",
            "sampled_row_applications", "extrapolated_row_applications")


@torch.no_grad()
def anticipate(optimizer):
    proposals = {}
    staged = []
    counts = dict.fromkeys(COUNTERS, 0)
    counts["steps"] = 1
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("optimism requires finite raw gradients")
            history = optimizer.state.get(parameter, {})
            previous = history.get(PREVIOUS)
            counts["parameter_applications"] += 1
            if group["algorithm"] == "rownorm":
                rows = (optimizer._sampled_rows[parameter] if group["sampled_rows_required"]
                        else torch.arange(len(parameter), device=parameter.device))
                seen = history.get(SEEN, torch.zeros(len(parameter), dtype=torch.bool,
                                                     device=parameter.device))
                forecast = torch.zeros_like(gradient)
                selected = gradient[rows]
                valid = seen[rows]
                if previous is not None:
                    forecast[rows] = torch.where(valid[:, None], 2 * selected - previous[rows], selected)
                else:
                    forecast[rows] = selected
                next_previous = (torch.zeros_like(gradient) if previous is None else previous.clone())
                next_previous[rows] = selected
                next_seen = seen.clone()
                next_seen[rows] = True
                count = int(valid.sum())
                counts["sampled_row_applications"] += len(rows)
                counts["extrapolated_row_applications"] += count
                counts["extrapolated_applications"] += int(count > 0)
                staged.append((parameter, next_previous, next_seen))
            else:
                forecast = gradient if previous is None else 2 * gradient - previous
                counts["extrapolated_applications"] += int(previous is not None)
                staged.append((parameter, gradient.detach().clone(), None))
            if not bool(torch.isfinite(forecast).all()):
                raise ValueError("nonfinite optimistic gradient forecast")
            proposals[parameter] = forecast
    # All forecasts validate before any history is changed. No draws/forwards.
    for parameter, previous, seen in staged:
        optimizer.state[parameter][PREVIOUS] = previous
        if seen is not None:
            optimizer.state[parameter][SEEN] = seen
    for key, count in counts.items():
        optimizer.optimism_stats[key] += count
    return proposals


def validate_history(state, parameter, *, rownorm):
    previous = state.get(PREVIOUS)
    if (not isinstance(previous, torch.Tensor) or previous.shape != parameter.shape
            or previous.dtype != parameter.dtype or not bool(torch.isfinite(previous).all())):
        raise ValueError("invalid optimism raw-gradient history")
    if rownorm:
        seen = state.get(SEEN)
        if (not isinstance(seen, torch.Tensor) or seen.dtype != torch.bool
                or tuple(seen.shape) != (len(parameter),)):
            raise ValueError("invalid optimism sampled-row history")
        if bool((previous[~seen] != 0).any()):
            raise ValueError("unseen optimistic rows contain history")
