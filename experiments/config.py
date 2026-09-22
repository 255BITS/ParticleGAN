"""Repository config I/O; public ParticleGAN constructors accept ordinary dicts."""
from pathlib import Path


def read_config(path):
    """Read a TOML or historical YAML mapping, with string keys."""
    path = Path(path)
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10 experiment extra
            import tomli as tomllib
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    elif path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        with path.open() as stream:
            config = yaml.safe_load(stream)
        if config is None:
            config = {}
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}; use .toml, .yaml, or .yml")
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        raise ValueError("config must be a mapping with string keys")
    return config


def recipe_defaults(name):
    """Translate public recipe fields to the existing flat experiment schema.

    Kept separate from parser imports so runner stubs do not need Torch. Legacy
    research options stay in each trainer, and explicit config values win.
    """
    from particlegan import get_recipe

    recipe = get_recipe(name)
    keys = ("batch_size", "z_dim", "num_particles", "lr", "d_lr_mult",
            "loss_type", "gan_mode", "reg_arm", "reg_coeff", "reg_every",
            "reg_method", "lr_anneal_start", "lr_floor")
    defaults = {key: getattr(recipe, key) for key in keys}
    defaults["beta1"] = recipe.betas[0]
    if name == "100gaussians":
        defaults.update(beta2=recipe.betas[1], epochs=recipe.total_steps // 1000, steps_per_epoch=1000,
                        lambda_ep=recipe.prior_reg, ema_decay=recipe.ema_decay, reg_kappa=recipe.reg_kappa)
    elif name == "denoising":
        defaults.update(model=recipe.model, d_mode=recipe.conditioning,
                        classes=recipe.num_classes, ucd_target=recipe.ucd_target,
                        ucd_lambda=recipe.ucd_weight, alpha_bar=list(recipe.alpha_bar),
                        steps=recipe.total_steps, prior_lr_mult=recipe.prior_lr_mult,
                        prior_reg=recipe.prior_reg, reg_kappa=recipe.reg_kappa,
                        ema=recipe.ema_decay)
    return defaults


def merge_config(defaults, user):
    """Resolve experiment defaults consistently for direct and grid launches.

    CIFAR condition caching defaults to enabled only for pretrained critics.
    Infer this before the runner writes its fully resolved config, otherwise
    historical pixel-critic files inherit an unsupported optimization. Explicit
    values remain untouched and are checked by the trainer's validation.
    """
    config = {**defaults, **user}
    if ("d_backbone" in defaults and "cache_condition" in defaults
            and "cache_condition" not in user):
        config["cache_condition"] = config["d_backbone"] in (
            "pretrained_resnet18", "pretrained_resnet34")
    return config
