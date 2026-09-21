"""Paired control continuation taken from the model-glue winner.

The SD1.5 to SANA text bridge at code revision
f67054d329fe5246510a094dd791d485526e3dd6 kept a 0.1 paired embedding anchor and a
functional match through a frozen recipient, trained only the output head, and
left GAN / b_cap configured but inactive (adv weight 0). A repaired b_cap
screen with adversarial weight 0.001 did not win validation. This module is the
shared objective for the 2D CPU toy and the Lunar Lander particle continuation.
"""
import torch
from torch.nn import functional as F

ANCHOR_WEIGHT = 0.1
FUNCTIONAL_WEIGHT = 1.0
GUIDED_WEIGHT = 0.05
GUIDANCE = 4.5
ADV_WEIGHT = 0.0
HEAD_LR = 1e-5
EMA_DECAY = 0.98
MAX_GRAD_NORM = 1.0
TRAINABLE_PARTS = "heads"
BETAS = (0.0, 0.999)
DT = 0.2


def kinematic_response(state, action):
    """One double-integrator step. State is [x, y, vx, vy]; action is [ax, ay].

    Linear in the action, so matching the frozen recipient is a scaled paired
    action error. It is not a Lunar Lander simulator.
    """
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError("kinematic state must be [batch, 4]")
    if action.ndim != 2 or action.shape[1] != 2 or action.shape[0] != state.shape[0]:
        raise ValueError("kinematic action must be [batch, 2] aligned with state")
    velocity = state[:, 2:4] + DT * action
    position = state[:, :2] + DT * velocity
    return torch.cat([position, velocity], 1)


def glue_objective(student_action, expert_action, student_response, teacher_response, empty_response):
    """0.1 normalized paired anchor plus the model-glue functional response.

    Both empty branches share ``empty_response`` (native unconditional
    conditioning in model-glue). Teacher tensors are detached. The denominator
    is detached recipient prompt-effect power.
    """
    if student_action.shape != expert_action.shape:
        raise ValueError("student and expert actions must share a shape")
    if student_response.shape != teacher_response.shape or student_response.shape != empty_response.shape:
        raise ValueError("student, teacher, and empty responses must share a shape")
    teacher_response = teacher_response.detach()
    empty_response = empty_response.detach()
    variance = expert_action.detach().var(dim=0, unbiased=False).clamp_min(1e-3)
    anchor = ((student_action - expert_action).pow(2) / variance).mean()
    positive = F.mse_loss(student_response, teacher_response)
    empty = F.mse_loss(empty_response, empty_response)
    guided_student = empty_response + GUIDANCE * (student_response - empty_response)
    guided_teacher = empty_response + GUIDANCE * (teacher_response - empty_response)
    guided = F.mse_loss(guided_student, guided_teacher)
    power = (teacher_response - empty_response).pow(2).mean().clamp_min(1e-5)
    functional = (positive + empty + GUIDED_WEIGHT * guided) / power
    loss = ANCHOR_WEIGHT * anchor + FUNCTIONAL_WEIGHT * functional
    if ADV_WEIGHT != 0:
        raise RuntimeError("This continuation keeps adversarial weight at 0")
    return loss, dict(action_anchor=anchor, functional=functional, response_power=power)
