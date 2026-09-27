"""Verbatim frozen image model/data/scorer definitions; no legacy learner imports."""
import math
import torch
from torch import nn
import torch.nn.functional as F
MIN_STABLE_CHECKS = 5

def templates(spec):
    """No labels are supplied during training; these templates only define data."""
    kind = spec["pattern"]
    images = torch.zeros(spec["modes"], 1, 8, 8)
    if kind == "stripes2":
        images[0, 0, 3:5, :] = 1.
        images[1, 0, :, 3:5] = 1.
    elif kind in ("bars4", "bars8"):
        positions = [1, 5] if kind == "bars4" else [0, 2, 4, 6]
        for index, position in enumerate(positions):
            images[index, 0, :, position:position + 2] = 1.
            images[index + len(positions), 0, position:position + 2, :] = 1.
    elif kind == "blobs4":
        for index, (row, col) in enumerate(((1, 1), (1, 5), (5, 1), (5, 5))):
            images[index, 0, row:row + 2, col:col + 2] = 1.
    elif kind == "intensity2":
        images[0, 0, 2:6, 2:6] = .35
        images[1, 0, 2:6, 2:6] = .85
    else:
        raise ValueError(f"unknown pattern {kind}")
    return images


class Generator(nn.Module):
    def __init__(self, spec):
        super().__init__()
        width = spec["width"]
        self.architecture = spec["architecture"]
        self.input = nn.Linear(spec["z_dim"], width * 4)
        if self.architecture == "residual_upsample":
            self.first = nn.Conv2d(width, width, 3, padding=1)
            self.second = nn.Conv2d(width, width, 3, padding=1)
        else:
            self.first = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
            self.second = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.output = nn.Conv2d(width, 1, 3, padding=1)
        self.width = width

    def forward(self, z):
        value = F.leaky_relu(self.input(z).reshape(-1, self.width, 2, 2), .2)
        if self.architecture == "residual_upsample":
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            value = value + F.leaky_relu(self.first(value), .2)
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            value = value + F.leaky_relu(self.second(value), .2)
        else:
            value = F.leaky_relu(self.first(value), .2)
            value = F.leaky_relu(self.second(value), .2)
        value = self.output(value).sigmoid()
        if self.architecture == "uniform_generator":
            value = value.mean((2, 3), keepdim=True).expand(-1, -1, 8, 8)
        return value


class Discriminator(nn.Module):
    def __init__(self, spec):
        super().__init__()
        # The deliberately tiny-generator diagnostic keeps a healthy D.
        width = max(12, spec["width"])
        self.mean_only = spec["architecture"] == "mean_discriminator"
        self.network = nn.Sequential(nn.Conv2d(1, width, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                                     nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                                     nn.Flatten(), nn.Linear(8 * width, 1))

    def forward(self, images):
        if self.mean_only:
            images = images.mean((2, 3), keepdim=True).expand(-1, -1, 8, 8)
        return self.network(images).flatten()


def image_metrics(images, centers, thresholds):
    if images.ndim != 4 or images.shape[1:] != (1, 8, 8) or not torch.isfinite(images).all():
        raise ValueError("finite N×1×8×8 images required")
    rmses = (images[:, None] - centers[None]).square().mean((2, 3, 4)).sqrt()
    nearest_rmse, assignment = rmses.min(1)
    quality = nearest_rmse <= thresholds["quality_rmse"]
    counts = torch.bincount(assignment[quality], minlength=len(centers))
    fractions = counts.double() / len(images)
    all_counts = torch.bincount(assignment, minlength=len(centers)).double() / len(images)
    return dict(modes=int((fractions >= thresholds["min_mode_fraction"]).sum()),
                hq=float(quality.double().mean()), mean_rmse=float(nearest_rmse.mean()),
                quality_mode_fractions=fractions.tolist(),
                mode_fractions=all_counts.tolist(),
                distribution_tv=float((all_counts - 1. / len(centers)).abs().sum() / 2))


def evaluation_steps(spec):
    count = spec["thresholds"]["observations"]
    if spec["steps"] < count:
        raise ValueError("budget must permit all 24 distinct evaluations")
    return [math.ceil(index * spec["steps"] / count) for index in range(1, count + 1)]


def sustained(curve, requirements, *, expected_steps, minimum=MIN_STABLE_CHECKS):
    """Find a passing suffix, never count a transient pass as convergence.

    Only recorded observations are certified. First/confirmation times include
    setup and measurement overhead, and are not inferred for historical rows.
    """
    def passes(point):
        for key, op, bound in requirements:
            value = point.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return False
            if not (value >= bound if op == ">=" else value <= bound):
                return False
        return True
    if minimum < 2:
        raise ValueError("minimum must be at least two observations")
    if any(b["step"] <= a["step"] for a, b in zip(curve, curve[1:])):
        raise ValueError("observations must have unique increasing steps")
    passing = [passes(p) for p in curve]
    complete = [p["step"] for p in curve] == sorted(expected_steps)
    first = next((p for p, ok in zip(curve, passing) if ok), None)
    start = len(curve)
    while start and passing[start - 1]:
        start -= 1
    suffix = curve[start:]
    stable = complete and len(suffix) >= minimum
    return {"complete": complete, "observations": len(curve), "passing_observations": sum(passing),
            "minimum_stable_checks": minimum, "passing_suffix": len(suffix),
            "first_pass_step": first["step"] if first else None,
            "stable_from_step": suffix[0]["step"] if stable else None,
            "stable_from_seconds": suffix[0].get("seconds") if stable else None,
            "confirmed_step": suffix[minimum - 1]["step"] if stable else None,
            "confirmed_seconds": suffix[minimum - 1].get("seconds") if stable else None}
