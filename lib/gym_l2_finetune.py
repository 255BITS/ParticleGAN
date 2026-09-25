"""L2 controller finetune objective: standardized expert action MSE only.

This is the imitation fine-tune loss. It does not include reconstruction,
contact BCE, prior spread, or any discriminator term. Those belong to other
arms and must not be added here.
"""
from torch.nn import functional as F


LOCKED_SUPERVISION = dict(
    primary="standardized_action_mse",
    terms=["standardized_action_mse"],
    adversarial=False,
    contact_bce=False,
    frozen=["G1", "G3", "E", "prior", "D"],
    trainable=["E_control", "G2"],
)


def l2_objective(scaler, predicted_action, standardized_target):
    """Mean squared error between the standardized command and the expert command."""
    loss = F.mse_loss(scaler.action(predicted_action), standardized_target)
    return loss, {"standardized_action_mse": loss}


def assert_l2_supervision(config, summary):
    """Reject a checkpoint or run whose recorded objective is not action MSE."""
    if config.get("arm") != "l2":
        raise ValueError("L2 finetune requires arm=l2")
    if config.get("supervision") != LOCKED_SUPERVISION:
        raise ValueError("L2 finetune supervision changed; refusing extra or adversarial loss terms")
    if summary.get("loss_terms") != ["standardized_action_mse"]:
        raise ValueError("L2 finetune loss terms must be standardized action MSE only")
    if summary.get("adversarial_updates") != 0 or summary.get("discriminator_optimizer_record_draws") != 0:
        raise ValueError("L2 finetune recorded discriminator updates")
    if summary.get("simulator_calls") != 0:
        raise ValueError("L2 finetune must not call the simulator")
