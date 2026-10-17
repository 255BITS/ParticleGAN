"""Source-bound original common26 diagnostic case using Forge's supervisor.

This module is inert on import. ROOT calls preparation, admission, retained-byte
certification and publication only in its authorized cumulative metadata phase.
An ordinary numerical FAIL does not qualify its dependencies or stop independent
diagnostic cases. Invalid/interrupted physical attempts halt the campaign.
"""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
from importlib.machinery import NamespaceLoader
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

MODULE = "experiments.forge.canonical_full_atlas_case"
MEMBER = "experiments/forge/canonical_full_atlas_case.py"
SCHEMA = "pg_common26_full_atlas_diagnostic_case_v1"
CONFIG = "configs/100gaussians/atlas.json"
PROTOCOL = "configs/forge/protocols/screening.json"
HOST_MEMORY_MB = 2048
GPU_MEMORY_MB = 2048
DEVICE = "cuda:1"
VISIBILITY = "0,1"
IMAGE_TASKS = ("img_stripes2", "img_bars4", "img_blobs4", "img_intensity2")
MOG_TASKS = ("ring16_acquisition", "mode_hold", "vector_two_broad",
             "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic",
             "vector_overlap", "vector_spiral", "grid100", "rotated100",
             "staggered100", "ring_hold", "ring_extension")
AE_TASKS = ("ae_gan_hold",)
_PREREQUISITE_CACHE = {}


def _utils():
    from .contracts import atomic_json, file_hash, read_json, stable_hash
    return atomic_json, file_hash, read_json, stable_hash


def _base(request):
    return {key: deepcopy(value) for key, value in request.items()
            if key not in {"admission", "target", "command"}}


def case_id(task_id):
    return "canonical-common26-full-atlas-" + task_id + "-ember561-v1"


def adapter_module(task_id):
    if task_id == "ring_extension":
        raise ValueError("ring extension is a retained consumer of the uninterrupted hold group, never a fresh case")
    if task_id in IMAGE_TASKS:
        return "experiments.forge.canonical_image_adapter"
    if task_id in MOG_TASKS:
        return "experiments.forge.canonical_mog_adapter"
    if task_id in AE_TASKS:
        return "experiments.forge.canonical_ae_adapter"
    raise ValueError("this source-bound owner does not support the canonical task")


def _device(value, selected=None):
    task = value.get("task", value)
    actual = selected or value.get("runtime", {}).get("device")
    if task["id"] in AE_TASKS:
        if actual not in {None, "cpu"}:
            raise ValueError("the original AE stays CPU1 with no GPU allocation")
        return "cpu"
    actual = actual or DEVICE
    if actual not in {"cuda:0", "cuda:1"}:
        raise ValueError("only the declared physical GPU0/GPU1 pair is authorized")
    return actual


def _visibility(value):
    return "" if _device(value) == "cpu" else VISIBILITY


def _base_protocol(protocol):
    protocol = deepcopy(protocol)
    protocol.pop("scientific_repeat", None)
    return protocol


def dependency_status(task, outcomes, *, request=None):
    """Numerical failure is not transferable prerequisite or checkpoint credit."""
    for dependency in task.get("dependencies", []):
        parent = outcomes.get(dependency["task"])
        if request is None or not parent or parent.get("status") != "PASS" or parent.get("certified") is not True:
            return {"status": "BLOCKED", "reason": "canonical prerequisite has no certified PASS",
                    "dependency": deepcopy(dependency)}
        parent_request = parent.get("request", {})
        _, _, _, digest = _utils()
        if (parent_request.get("schema") != SCHEMA or parent_request.get("case_id") != case_id(dependency["task"])
                or parent_request.get("task", {}).get("id") != dependency["task"]
                or parent_request.get("source", {}).get("digest") != request["source"]["digest"]
                or digest(parent_request.get("candidate")) != digest(request["candidate"])
                or digest(_base_protocol(parent_request.get("protocol", {}))) != digest(_base_protocol(request["protocol"]))
                or digest(parent_request.get("binding", {}).get("recipe")) != digest(request["binding"]["recipe"])
                or parent.get("certificate", {}).get("full_protocol_complete") is not True
                or parent.get("certificate", {}).get("grade", {}).get("status") != "PASS"):
            return {"status": "BLOCKED", "reason": "prerequisite is not the accepted same-source full-original campaign case",
                    "dependency": deepcopy(dependency)}
        if dependency["kind"] == "checkpoint" and not parent.get("checkpoint_binding"):
            return {"status": "BLOCKED", "reason": "own certified checkpoint binding is unavailable",
                    "dependency": deepcopy(dependency)}
    return {"status": "READY"}


def _pin(path):
    _, file_hash, _, _ = _utils()
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def _read_pin(pin):
    _, file_hash, read_json, _ = _utils()
    path = Path(pin["path"])
    if file_hash(path) != pin["sha256"] or path.stat().st_size != pin["bytes"]:
        raise ValueError("bound prerequisite bytes changed: " + str(path))
    return read_json(path)


def bind_mode_hold_prerequisite(packet):
    """Paid parent recertification from its own actual completed physical attempt."""
    _, _, read_json, _ = _utils()
    if (packet.get("schema") != SCHEMA or packet["request"]["task"]["id"] != "mode_hold"
            or packet["rows"][0].get("status") != "PASS" or packet["rows"][0].get("certified") is not True):
        raise ValueError("the current campaign's accepted mode_hold packet is required")
    attempt = packet["rows"][0]["attempt_key"]
    target = Path(packet["output"]) / "mode_hold"
    terminal_path = Path(packet["queue_root"]) / "policy/attempts" / attempt / "supervisor-terminal.json"
    terminal = read_json(terminal_path)
    admitted = read_json(target / "request.json")["admission"]
    certificate = certify(packet, target, terminal, admitted["token"])
    if certificate["grade"]["status"] != "PASS":
        raise ValueError("mode_hold must independently recertify PASS before ring admission")
    return {"schema": "pg_common26_actual_mode_hold_prerequisite_v1", "parent_packet": deepcopy(packet),
            "target": str(target.resolve()), "attempt_key": attempt, "certificate": certificate,
            "terminal_pin": _pin(terminal_path),
            "files": {name: _pin(target / name) for name in
                      ("raw-result.json", "grade.json", "attestation.json", "INITIALIZATION.json", "MODEL_STARTED.json")},
            "frozen_request_pin": deepcopy(packet["frozen_request_pin"]),
            "terminal_token_sha256": hashlib.sha256(admitted["token"].encode()).hexdigest()}


def verify_ring_prerequisite(request):
    """No PASS checkbox: pin actual artifact/terminal provenance and frozen grade."""
    if request["task"]["id"] != "ring_hold":
        if request.get("prerequisites"):
            raise ValueError("this independent task cannot carry another case's gate credit")
        return
    _, _, _, digest = _utils()
    proof = request.get("prerequisites", {}).get("mode_hold")
    if not isinstance(proof, dict) or proof.get("schema") != "pg_common26_actual_mode_hold_prerequisite_v1":
        raise ValueError("ring endurance requires its current actual mode_hold artifact/terminal proof")
    parent = proof["parent_packet"]
    outcomes = {"mode_hold": {**parent["rows"][0], "request": parent["request"]}}
    if dependency_status(request["task"], outcomes, request=request)["status"] != "READY":
        raise ValueError("ring prerequisite is not the same-source full-original campaign PASS")
    if request["protocol"]["scientific_repeat"].get("prerequisite_proof_sha256") != digest(proof):
        raise ValueError("ring repeat is not bound to its actual prerequisite proof")
    target = Path(parent["output"]) / "mode_hold"
    expected_terminal = Path(parent["queue_root"]) / "policy/attempts" / proof["attempt_key"] / "supervisor-terminal.json"
    if (proof["target"] != str(target.resolve()) or proof["attempt_key"] != parent["rows"][0]["attempt_key"]
            or proof["terminal_pin"]["path"] != str(expected_terminal.resolve())
            or proof["frozen_request_pin"] != parent["frozen_request_pin"]):
        raise ValueError("ring gate proof references another physical attempt")
    terminal = _read_pin(proof["terminal_pin"])
    if (terminal.get("attempt_status") != "completed"
            or hashlib.sha256(terminal.get("token", "").encode()).hexdigest() != proof["terminal_token_sha256"]):
        raise ValueError("ring gate lacks its matching maintained completed terminal")
    expected_names = {"raw-result.json", "grade.json", "attestation.json", "INITIALIZATION.json", "MODEL_STARTED.json"}
    if set(proof.get("files", {})) != expected_names:
        raise ValueError("ring prerequisite artifact proof is incomplete")
    for name, pin in proof["files"].items():
        if pin["path"] != str((target / name).resolve()):
            raise ValueError("ring gate points to a foreign retained artifact")
        _read_pin(pin)
    if _read_pin(proof["frozen_request_pin"]) != parent["request"]:
        raise ValueError("ring gate frozen request differs from its accepted parent")
    key = digest(proof)
    if key not in _PREREQUISITE_CACHE:
        certificate = certify(parent, target, terminal, terminal["token"])
        if certificate != proof["certificate"] or certificate["grade"]["status"] != "PASS":
            raise ValueError("the bound prior gate did not independently recertify PASS")
        _PREREQUISITE_CACHE[key] = deepcopy(certificate)


def validate_request(request):
    from .canonical_full_atlas_repeat import validate_scientific_repeat
    _, _, _, digest = _utils()
    task = request["task"]
    if request.get("schema") != SCHEMA or request.get("case_id") != case_id(task["id"]):
        raise ValueError("unknown canonical diagnostic case")
    adapter_module(task["id"])
    runtime = request["runtime"]
    device = _device(request); gpus = int(device != "cpu"); memory_mb = GPU_MEMORY_MB if gpus else 0
    if (runtime.get("device") != device or runtime.get("cuda_visible_devices") != _visibility(task)
            or runtime.get("torch_threads") != 1 or runtime.get("gpus") != gpus
            or runtime.get("deterministic") is not True or runtime.get("tf32") is not False
            or runtime.get("dtype") != "float32"
            or runtime.get("gpu_memory_limit_mb") != memory_mb):
        raise ValueError("the exact task-owned CPU1/float32/device compute contract is required")
    resources = task["resources"]
    if (type(resources.get("timeout_seconds")) is not int
            or resources["timeout_seconds"] not in {300, 900, 1800, 3600}
            or resources.get("gpus") != gpus or resources.get("cpu_threads") != 1
            or resources.get("gpu_memory_mb") != memory_mb
            or (not gpus and resources.get("allow_cpu") is not True)):
        raise ValueError("the canonical resource declaration changed")
    diagnostic = request.get("diagnostic")
    if diagnostic != {"original_registered_view": 3, "original_task_slots": 26,
                      "continue_independent_after_numeric_fail": True,
                      "qualification_credit": False, "default_adoption": False,
                      "speed_ranking": False}:
        raise ValueError("diagnostic continuation cannot become family qualification")
    source = request["source"]
    if digest(source["files"]) != source["digest"]:
        raise ValueError("invalid execution-source manifest")
    return validate_scientific_repeat(request)


def source_guard(request, root):
    from .sources import verify_snapshot
    _, _, read_json, _ = _utils()
    validate_request(request)
    root = Path(root).resolve()
    manifest = {key: value for key, value in request["source"].items() if key != "snapshot_path"}
    if (root != Path(request["source"]["snapshot_path"]).resolve()
            or read_json(root / "forge-source.json") != manifest
            or Path(__file__).resolve() != root / MEMBER):
        raise ValueError("only the actual copied-source controller may execute")
    verify_snapshot(root, manifest)
    verify_ring_prerequisite(request)
    for name, module in tuple(sys.modules.items()):
        if (name.split(".", 1)[0] not in {"particlegan", "benchmarks", "experiments", "lib"}
                and name != "_forge_original_convergence_gate") or module is None:
            continue
        loaded = getattr(module, "__file__", None)
        namespaces = tuple(getattr(module, "__path__", ()))
        if loaded is None and (not namespaces or not isinstance(
                getattr(getattr(module, "__spec__", None), "loader", None), NamespaceLoader)):
            raise ValueError("missing-file scientific module is not an owned namespace: " + name)
        if loaded is not None:
            member = Path(loaded).resolve()
            if not member.is_relative_to(root) or member.relative_to(root).as_posix() not in manifest["files"]:
                raise ValueError("foreign already-imported scientific module: " + name)
        if any(not Path(namespace).resolve().is_relative_to(root) for namespace in namespaces):
            raise ValueError("foreign scientific package namespace: " + name)


def compile_owner_source(adapter, root, task_id):
    """Compile the exact inert owner seam; do not require an invented AE API."""
    if task_id not in AE_TASKS:
        return adapter.compile_source_preflight(root)
    for relative in adapter.SOURCE_PINS:
        adapter._source(root, relative)
    overlay = adapter.derive_host_train(adapter._source(root, adapter.HOST_PATH).decode())
    compile(overlay, "<metadata-only-original-AE-objective-overlay>", "exec")
    return {"schema": "pg_common26_original_ae_compile_preflight_v1", "source_files": len(adapter.SOURCE_PINS),
            "model_constructors": 0, "forwards": 0, "updates": 0, "evaluation_draws": 0}


def preflight(request, root):
    """Copied-source compile and exact declarations only; no model imports."""
    _, _, _, digest = _utils()
    source_guard(request, root)
    adapter = importlib.import_module(adapter_module(request["task"]["id"]))
    binding = adapter.resolve_binding(root, request["candidate"], request["task"], request["protocol"])
    if digest(binding) != digest(request["binding"]):
        raise ValueError("copied binding differs from the preregistered binding")
    compile_owner_source(adapter, root, request["task"]["id"])
    if any(name == "torch" or name.startswith("torch.") or name == "particlegan"
           or name.startswith("particlegan.") for name in sys.modules):
        raise ValueError("copied metadata preflight imported a scientific model package")
    return {"schema": "pg_common26_full_atlas_copied_preflight_v1", "status": "PASS",
            "frozen_request_digest": digest(_base(request)), "source_digest": request["source"]["digest"],
            "repeat": deepcopy(request["protocol"]["scientific_repeat"]),
            "source_contract_sha256": binding["source_contract_sha256"],
            "model_constructors": 0, "forwards": 0, "updates": 0, "evaluation_draws": 0}


def gpu_fit(device=DEVICE):
    """Paid capacity probe of the exact selected physical card; no process changes."""
    if device not in {"cuda:0", "cuda:1"}:
        raise ValueError("only the declared physical GPU pair can be probed")
    index = int(device.split(":")[1])
    row = subprocess.check_output(
        ["nvidia-smi", "--id=" + str(index), "--query-gpu=index,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip().split(",")
    if len(row) != 4 or row[0].strip() != str(index):
        raise ValueError("the selected physical card capacity could not be resolved")
    total, used = int(row[2].strip()), int(row[3].strip())
    if total - used < 2 * GPU_MEMORY_MB:
        raise ValueError("the selected card lacks its allocation plus preserved headroom")
    return {"physical_gpu_index": index, "model": row[1].strip(), "memory_total_mb": total,
            "memory_used_mb": used, "memory_available_mb": total - used,
            "declared_memory_limit_mb": GPU_MEMORY_MB}


def admission_guard(request):
    from .queue import lease_held, process_identity
    _, _, read_json, _ = _utils()
    admission = request["admission"]
    allowance = request["task"]["resources"]["timeout_seconds"]
    directory = Path(admission["lease_paths"][-1]).parent
    supervisor = read_json(directory / "supervisor-request.json")
    child = read_json(directory / "child.json")
    if (supervisor.get("token") != admission["token"] or child.get("token") != admission["token"]
            or child.get("pid") != os.getpid() or child.get("process_identity") != process_identity(os.getpid())
            or supervisor.get("command") != request["command"]
            or supervisor.get("source") != request["source"]
            or supervisor.get("started_monotonic") != admission["started_monotonic"]
            or supervisor.get("deadline_monotonic") != admission["deadline_monotonic"]
            or child.get("deadline_monotonic") != admission["deadline_monotonic"]
            or time.monotonic() >= admission["deadline_monotonic"]
            or admission["deadline_monotonic"] - admission["started_monotonic"] != allowance):
        raise ValueError("actual maintained child/admission token/deadline does not match")
    if len(admission["lease_fds"]) != 2 or len(admission["lease_paths"]) != 2:
        raise ValueError("exact study and physical execution leases are required")
    for fd, path in zip(admission["lease_fds"], admission["lease_paths"], strict=True):
        inherited, declared = os.fstat(fd), os.stat(path)
        if (inherited.st_dev, inherited.st_ino) != (declared.st_dev, declared.st_ino) or not lease_held(Path(path)):
            raise ValueError("the inherited physical execution lease is not owned")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != _visibility(request):
        raise ValueError("task-owned physical placement differs from the frozen visibility")


def _json_tensors(value):
    """Serialize detached observation/state values without executing a model."""
    if isinstance(value, dict):
        return {key: _json_tensors(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_tensors(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def media_metric_groups(task):
    """Map declared gates to already recorded scalar values, without scoring."""
    evaluation = task["evaluation"]
    if evaluation["kind"] != "native_accuracy":
        return [{"title": "Original scored metrics", "normalized": False,
                 "metrics": deepcopy(evaluation.get("thresholds", []))}]
    coverage = []
    for key, bound in evaluation["coverage_thresholds"].items():
        if key in {"min_samples", "all_finite"}:
            continue
        name = {"min_modes": "modes", "min_precision": "precision", "max_mass_tv": "mass_tv"}.get(key, key)
        coverage.append([name, ">=" if key.startswith("min_") else "<=", bound])
    accuracy = [["accuracy." + key, "<=", bound] for key, bound in evaluation["accuracy_limits"].items()]
    return [{"title": "Coverage: metric / declared bound", "normalized": True, "metrics": coverage},
            {"title": "Accuracy: metric / declared bound", "normalized": True, "metrics": accuracy}]


def media_value(row, path):
    value = row
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if type(value) in (int, float) and math.isfinite(value) else None


def goal_caption(task, projection, step):
    grades = projection["task_grades"]
    verdict = ("final gates: hold=" + grades["ring_hold"]["status"] + "; immediate extension=" + grades["ring_extension"]["status"]
               if task["id"] == "ring_hold" else "final full-protocol verdict " + projection["grade"]["status"])
    return f"Full original Atlas · {task['id']} · update {step} · {verdict}"


def _media(target, states, observations, task, projection):
    """Every frame uses an actual scored observation; no extra sampling."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from PIL import Image
    import numpy as np
    observed = {row["step"]: row for row in observations}
    clocks = [state["step"] for state in states]
    if (not states or len(observed) != len(observations) or clocks != sorted(set(clocks))
            or any(step not in observed for step in clocks)):
        raise ValueError("goal states must join ordered original scored observations")
    metric_groups = media_metric_groups(task)
    image = np.asarray(states[0]["generated"]).ndim == 4
    limits = None
    if not image:
        clouds = [np.asarray(state[key]).reshape(len(state[key]), -1)
                  for state in states for key in ("target", "generated", "reconstructed", "anchors") if key in state]
        if any(cloud.shape[1] != 2 or not np.isfinite(cloud).all() for cloud in clouds):
            raise ValueError("vector goal media requires finite original two-dimensional outputs")
        union = np.concatenate(clouds, axis=0)
        low, high = union.min(axis=0), union.max(axis=0)
        padding = np.maximum(.05 * (high - low), .1)
        limits = [(low[i] - padding[i], high[i] + padding[i]) for i in range(2)]
    def cloud_panel(ax, reference, generated, *, title, anchors=None):
        dense = len(reference) > 128
        ax.scatter(reference[:, 0], reference[:, 1], s=3 if dense else 40, alpha=.13 if dense else .8,
                   marker="." if dense else "x", color="#159c85",
                   label="Retained target draw" if dense else "Declared target")
        ax.scatter(generated[:, 0], generated[:, 1], s=4, alpha=.25, color="#6741b3", label="Measured live outputs")
        if anchors is not None:
            ax.scatter(anchors[:, 0], anchors[:, 1], s=70, marker="x", color="#333333", label="Hold anchors")
        ax.set(xlim=limits[0], ylim=limits[1], title=title)
        ax.set_aspect("equal", adjustable="box"); ax.legend(fontsize=7)
    frames = []
    for state in states:
        observation = observed[state["step"]]
        generated, reference = np.asarray(state["generated"]), np.asarray(state["target"])
        if generated.ndim == 4 and generated.shape[1:] == (1, 8, 8):
            fig = plt.figure(figsize=(10, 5.5))
            grid = fig.add_gridspec(3, 1, height_ratios=(1, 2, 1.2))
            ax = fig.add_subplot(grid[0]); ax.axis("off")
            images = reference[:, 0] if reference.ndim == 4 else reference.reshape(-1, 8, 8)
            ax.imshow(np.concatenate(list(images), axis=1), cmap="gray", vmin=0, vmax=1)
            ax.set_title("Declared target templates")
            ax = fig.add_subplot(grid[1]); ax.axis("off")
            display = generated[:32, 0]
            rows = [np.concatenate(list(display[start:start + 8]), axis=1) for start in range(0, len(display), 8)]
            ax.imshow(np.concatenate(rows, axis=0), cmap="gray", vmin=0, vmax=1)
            ax.set_title("Original live outputs from all 32 prior centers")
            metric_axes = [fig.add_subplot(grid[2])]
        else:
            generated = generated.reshape(len(generated), -1)
            reference = reference.reshape(len(reference), -1)
            if task["id"] in AE_TASKS:
                fig, axes = plt.subplots(1, 3, figsize=(14, 4.2)); metric_axes = [axes[2]]
                reconstructed = np.asarray(state["reconstructed"]).reshape(len(state["reconstructed"]), -1)
                anchors = np.asarray(state["anchors"]).reshape(len(state["anchors"]), -1)
                cloud_panel(axes[0], reference, reconstructed, title="Live reconstruction: scheduled public noise")
                cloud_panel(axes[1], reference, generated, title="Independent prior output: scheduled public noise", anchors=anchors)
            else:
                fig, axes = plt.subplots(1, 1 + len(metric_groups), figsize=(5 * (1 + len(metric_groups)), 4.2))
                cloud_panel(axes[0], reference, generated, title=f"Retained scored output view ({len(generated)} points)")
                metric_axes = list(axes[1:])
        curve = [row for row in observations if row["step"] <= observation["step"]]
        for metric_ax, group in zip(metric_axes, metric_groups):
            plotted = set()
            for name, relation, bound in group["metrics"]:
                values = [media_value(row, name) for row in curve]
                if not any(value is not None for value in values):
                    continue
                divisor = bound if group["normalized"] else 1
                if type(divisor) not in (int, float) or divisor <= 0:
                    raise ValueError("normalized media requires a positive declared bound")
                label = f"{name} {relation} {bound}"
                if group["normalized"] or name not in plotted:
                    metric_ax.plot([row["step"] for row in curve],
                                   [np.nan if value is None else value / divisor for value in values],
                                   label=label if group["normalized"] else name)
                    plotted.add(name)
                if not group["normalized"]:
                    metric_ax.axhline(bound, linestyle=":", linewidth=1, label=label)
            if group["normalized"]:
                metric_ax.axhline(1., linestyle=":", color="black", linewidth=1, label="Declared bound = 1")
                metric_ax.set_yscale("symlog", linthresh=1)
            metric_ax.set(xlabel="Completed outer updates", ylabel=group["title"])
            if plotted:
                metric_ax.legend(fontsize=6 if group["normalized"] else 7, ncol=1 if group["normalized"] else 2)
        caption = goal_caption(task, projection, observation["step"])
        if task["evaluation"]["kind"] == "native_accuracy":
            caption += f"\n{task['evaluation']['eval_samples']:,} scored per read; final five plus independent {task['evaluation']['holdout_samples']:,} holdout"
        fig.suptitle(caption, fontsize=11)
        fig.tight_layout()
        buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=100); plt.close(fig)
        frames.append(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    frames[-1].save(target / "goal-final.png")
    frames[0].save(target / "goal.gif", save_all=True, append_images=frames[1:], duration=250, loop=0)


def _completed_steps(complete):
    """Actual owner state clock; never synthesize completion from intent."""
    clocks = [value["completed_steps"] for key in ("trainer", "policy")
              if isinstance(value := complete.get(key), dict) and "completed_steps" in value]
    if not clocks or any(type(clock) is not int or clock < 0 or clock != clocks[0] for clock in clocks):
        raise ValueError("the complete actual owner state has no consistent public clock")
    return clocks[0]


def execution_group(request, result):
    """One literal producer contract; an extension never creates another run."""
    _, _, _, digest = _utils()
    if request["task"]["id"] != "ring_hold":
        if request.get("grouped_tasks") or result.get("execution_group"):
            raise ValueError("an independent physical case cannot carry group credit")
        return None
    definitions = request.get("grouped_tasks", {})
    if (set(definitions) != {"ring_hold", "ring_extension"}
            or definitions["ring_hold"] != request["task"]
            or any(definitions[name].get("id") != name for name in definitions)):
        raise ValueError("the exact two original ring definitions are required")
    expected = request["binding"]["source_contract"]["grouped_execution"]
    if (expected.get("id") != "ring_endurance" or expected.get("producer_task_id") != "ring_hold"
            or expected.get("task_ids") != ["ring_hold", "ring_extension"]
            or expected.get("task_definitions") != definitions
            or expected.get("task_sha256") != {name: digest(task) for name, task in definitions.items()}
            or expected.get("physical_attempts") != 1 or expected.get("factory_calls") != 1
            or expected.get("allowance_seconds") != 3600
            or expected.get("independent_extension_replay") is not False
            or expected.get("checkpoint_restore") is not False):
        raise ValueError("the maintained single-attempt max-allowance group changed")
    continuity = result["evidence"].get("continuity", {})
    points = result["evidence"].get("dense", [])
    if (continuity.get("mode") != "uninterrupted" or not continuity.get("run_id")
            or continuity.get("resume_count") != 0
            or continuity.get("max_total_steps") != request["task"]["execution"]["max_total_steps"]
            or not points):
        raise ValueError("group evidence lacks one actual uninterrupted training run")
    group = {**deepcopy(expected),
             "prerequisite_proof_sha256": digest(request["prerequisites"]["mode_hold"]),
             "grouped_tasks_sha256": digest(definitions),
             "grouped_task_ids": ["ring_hold", "ring_extension"],
             "continuity": deepcopy(continuity), "completed_steps": points[-1]["step"],
             "shared_raw_evidence": True}
    if result.get("execution_group") != group:
        raise ValueError("actual group receipt differs from its source-bound producer and clocks")
    return group


def grade_projection(request, result, grader):
    """Unchanged task graders plus explicit prerequisite reduction for the consumer."""
    group = execution_group(request, result)
    if group is None:
        grade = grader(request["task"], result)
        return {"schema": "pg_common26_full_atlas_grade_projection_v1", "grade": grade,
                "task_grades": {request["task"]["id"]: grade},
                "raw_task_grades": {request["task"]["id"]: grade}, "execution_group": None}
    raw = {name: grader(request["grouped_tasks"][name], result)
           for name in ("ring_hold", "ring_extension")}
    projected = deepcopy(raw)
    if raw["ring_hold"]["status"] != "PASS":
        projected["ring_extension"] = {
            "status": "BLOCKED", "reason": "the same uninterrupted producer ring_hold did not pass",
            "dependency": {"task": "ring_hold", "kind": "checkpoint"},
            "raw_grade_preserved": deepcopy(raw["ring_extension"])}
    grade = (raw["ring_hold"] if raw["ring_hold"]["status"] != "PASS" else raw["ring_extension"])
    return {"schema": "pg_common26_full_atlas_grade_projection_v1", "grade": grade,
            "task_grades": projected, "raw_task_grades": raw, "execution_group": group}


def image_owner_health(request, result):
    """Ownership/finite-state acceptance is separate from original metric gates."""
    if request["task"]["id"] not in IMAGE_TASKS:
        return
    _, _, _, digest = _utils()
    evidence, applied = result["evidence"], result["applied"]
    guards = evidence.get("guards", {})
    counts = {name: 600 for name in ("generator", "prior", "discriminator", "noise")}
    lifecycle = {name: 600 for name in ("begin_step", "after_critic_step", "after_generator_backward",
                                      "after_generator_step", "finish_step")}
    audit = applied.get("lifecycle_audit", {})
    if (guards.get("all_finite") is not True or guards.get("public_policy_all_finite") is not True
            or guards.get("optimizer_updates") != counts
            or applied.get("optimizer_updates") != counts or applied.get("lifecycle_calls") != lifecycle
            or audit.get("owner") != "particlegan.UpdatePolicy" or audit.get("complete") is not True
            or audit.get("start_completed_steps") != 0 or audit.get("end_completed_steps") != 600
            or audit.get("observed_updates") != 600 or audit.get("calls") != lifecycle
            or audit.get("order_errors") != 0 or audit.get("pending") != []
            or audit.get("last_order") != list(lifecycle)
            or guards.get("hooks_exercised") is not True or guards.get("unintended_rng_deviations") != 0
            or digest(applied.get("recipe")) != digest(request["binding"]["recipe"])
            or applied.get("source_contract_sha256") != request["binding"]["source_contract_sha256"]
            or digest(applied.get("source_contract")) != request["binding"]["source_contract_sha256"]
            or applied.get("ordinary_observation_owner") != "live_generator_and_raw_prior_centers"
            or applied.get("evaluation_sampler_calls") != 0 or applied.get("evaluation_generator_calls") != 24
            or applied.get("output_noise_applied_to_metric") is not False
            or applied.get("dv12_applied_to_metric") is not False or applied.get("serial_backward") is not False):
        raise ValueError("original image owner health or full public control receipts are incomplete")
    purity = evidence.get("measurement_purity", [])
    if [point.get("step") for point in purity] != list(range(25, 601, 25)):
        raise ValueError("every original image observation requires its purity receipt")
    for point in purity:
        if (point.get("pure") is not True or point.get("before_sha256") != point.get("after_sha256")
                or point.get("global_before_sha256") != point.get("global_after_sha256")
                or point.get("named_rng", {}).get("unintended_rng_deviations") != 0
                or point.get("eval_draws") != 0 or point.get("forward_calls") != 1
                or point.get("original_metric_calls") != 1 or point.get("served_parameter_swap") is not False):
            raise ValueError("ordinary image observation changed its owned state/RNG/sampling law")


def component_owner_health(request, result, completed_steps):
    """Nonfinite state and missing policy ownership are invalid, not numeric FAIL."""
    if request["task"]["id"] in IMAGE_TASKS:
        image_owner_health(request, result)
        return
    _, _, _, digest = _utils()
    evidence = result["evidence"]; guards = evidence.get("guards", {})
    if (guards.get("all_finite") is not True or guards.get("hooks_exercised") is not True
            or guards.get("unintended_rng_deviations") != 0):
        raise ValueError("complete finite public owner and mechanism/RNG receipts are required")
    hooks = {name: completed_steps for name in ("begin_step", "after_critic_step",
             "after_generator_backward", "after_generator_step", "finish_step")}
    if request["task"]["id"] in MOG_TASKS:
        receipt = evidence.get("original_mog_owner", {}); audit = receipt.get("lifecycle", {})
        if (digest(result.get("original_mog_binding")) != digest(request["binding"])
                or digest(receipt.get("binding")) != digest(request["binding"])
                or digest(result.get("resolved_recipe")) != digest(request["binding"]["recipe"])
                or guards.get("optimizer_updates") != {name: completed_steps for name in ("generator", "discriminator", "prior")}
                or receipt.get("lifecycle_counts") != hooks
                or audit.get("owner") != "particlegan.UpdatePolicy" or audit.get("complete") is not True
                or audit.get("start_completed_steps") != 0 or audit.get("end_completed_steps") != completed_steps
                or audit.get("observed_updates") != completed_steps or audit.get("calls") != hooks
                or audit.get("order_errors") != 0 or audit.get("pending") != []
                or audit.get("last_order") != list(hooks)):
            raise ValueError("actual MoG source/Recipe/optimizer/public lifecycle ownership differs")
        observations = evidence.get("observations", evidence.get("dense"))
        purity = receipt.get("observations", [])
        if not observations or len(purity) < len(observations) or any(
                p.get("status") != "PURE" or not p.get("before_sha256")
                or p.get("training_draws_added") != 0 or p.get("optimizer_updates_added") != 0 for p in purity):
            raise ValueError("original MoG observations lack complete training-state purity")
        return
    if request["task"]["id"] in AE_TASKS:
        hooks = {name: completed_steps for name in ("begin_step", "before_critic_backward", "after_critic_step",
                 "before_generator_backward", "after_generator_backward", "after_generator_step", "finish_step")}
        applied = result.get("applied", {}); counts = {name: 250 for name in ("generator", "encoder", "prior", "discriminator", "noise")}
        if (completed_steps != 250 or guards.get("public_policy_state_finite") is not True
                or guards.get("optimizer_updates") != counts or applied.get("optimizer_updates") != counts
                or applied.get("lifecycle_calls") != hooks
                or digest(applied.get("recipe")) != digest(request["binding"]["recipe"])
                or applied.get("source_contract_sha256") != request["binding"]["source_contract_sha256"]
                or digest(applied.get("source_contract")) != request["binding"]["source_contract_sha256"]
                or applied.get("policy_owner") != "particlegan.UpdatePolicy" or applied.get("row_semantics") != "independent"
                or applied.get("observation_owner") != "live_G_E_and_fixed_width_actual_MoG"
                or applied.get("evaluation_DV12") is not False or applied.get("selected_or_averaged_metric") is not False
                or applied.get("external_horizon") != 250 or applied.get("intrinsic_horizon") is not None):
            raise ValueError("actual auxiliary AE full policy/objective/live-observation owner differs")
        purity = evidence.get("measurement_purity", [])
        clocks = [math.ceil(i * 250 / 24) for i in range(1, 25)]
        if ([p.get("step") for p in evidence.get("observations", [])] != clocks
                or [p.get("step") for p in purity] != [0, *clocks]
                or any(p.get("pure") is not True or p.get("before_sha256") != p.get("after_sha256")
                or p.get("global_rng_before_sha256") != p.get("global_rng_after_sha256")
                or p.get("unintended_rng_deviations") != 0 or p.get("scoring_weights") != "live"
                or p.get("evaluation_DV12") is not False for p in purity)):
            raise ValueError("ordinary AE evaluation mutated its owned state/RNG or sampling law")
        return
    raise ValueError("no source-bound complete-owner health contract for this task")


def child(path):
    from .canonical_full_atlas_repeat import FreshRepeatGuard
    atomic_json, file_hash, read_json, digest = _utils()
    path = Path(path).resolve(); raw_request = file_hash(path); request = read_json(path)
    root = Path(__file__).resolve().parents[2]; target = Path(request["target"])
    def guard_source():
        if file_hash(path) != raw_request:
            raise ValueError("admitted request changed")
        source_guard(request, root)
    guard_source(); admission_guard(request)
    from .sources import compute_profile, runtime_manifest
    actual = runtime_manifest()
    device = _device(request)
    actual_profile = (compute_profile("cpu", threads=1) if device == "cpu" else
                      compute_profile("cuda", request["runtime"]["compute_profile"]["model"], threads=1))
    if (any(request["runtime"].get(key) != value for key, value in actual.items())
            or request["runtime"]["compute_profile"] != actual_profile):
        raise ValueError("the actual child runtime differs from the frozen task cohort")
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device != "cpu":
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        profile = request["runtime"]["compute_profile"]
        if (properties.name != profile["model"]
                or f"{properties.major}.{properties.minor}" != profile["compute_capability"]):
            raise ValueError("actual selected physical GPU differs from the declared CUDA model cohort")
        torch.cuda.set_per_process_memory_fraction(
            GPU_MEMORY_MB * 1024 ** 2 / properties.total_memory, device=device)
    fresh = FreshRepeatGuard(request, source_guard=guard_source, admission_guard=lambda: admission_guard(request))
    construct = fresh.construct
    def record_start(factory, reader):
        owner = construct(factory, reader)
        atomic_json(target / "INITIALIZATION.json", fresh.require_owned(owner))
        atomic_json(target / "MODEL_STARTED.json", {
            "schema": "pg_common26_full_atlas_actual_start_v1", "initialized_monotonic": time.monotonic(),
            "frozen_request_digest": digest(_base(request)),
            "initialization_sha256": file_hash(target / "INITIALIZATION.json"),
            "models_constructed": True, "completed_updates": 0, "device": device, "gpus": int(device != "cpu"),
            "pid": os.getpid()})
        print(json.dumps({"event": "actual_model_start", "case": request["task"]["id"],
                          "seed": request["protocol"]["seed"], "device": device, "pid": os.getpid()}), flush=True)
        return owner
    fresh.construct = record_start
    adapter = importlib.import_module(adapter_module(request["task"]["id"]))
    kwargs = {"source_guard": guard_source, "fresh_repeat_guard": fresh}
    if request["task"]["id"] not in IMAGE_TASKS:
        kwargs.update(output_dir=target, device=device)
    result = adapter.run_case(root, request, **kwargs)
    states = _json_tensors(result.pop("retained_goal_states"))
    complete = result.pop("complete_state")
    completed_steps = _completed_steps(complete)
    atomic_json(target / "raw-result.json", result); atomic_json(target / "goal-states.json", states)
    torch.save(complete, target / "state.pt")
    from .views import grade_result
    projection = grade_projection(request, result, grade_result)
    grade = projection["grade"]
    atomic_json(target / "grade.json", grade)
    atomic_json(target / "grade-projection.json", projection)
    observations = result["evidence"].get("observations", result["evidence"].get("dense"))
    _media(target, states, observations, request["task"], projection)
    component_owner_health(request, result, completed_steps)
    guard_source(); admission_guard(request)
    reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 ** 2 if device != "cpu" else 0.0
    if reserved_mb > GPU_MEMORY_MB:
        raise ValueError("actual CUDA allocator exceeded the declared memory limit")
    names = ("INITIALIZATION.json", "MODEL_STARTED.json", "raw-result.json", "goal-states.json",
             "state.pt", "grade.json", "grade-projection.json", "goal.gif", "goal-final.png")
    files = {name: {"sha256": file_hash(target / name), "bytes": (target / name).stat().st_size} for name in names}
    guard_source(); admission_guard(request)
    atomic_json(target / "attestation.json", {
        "schema": "pg_common26_full_atlas_child_attestation_v1",
        "frozen_request_digest": digest(_base(request)), "source_digest": request["source"]["digest"],
        "repeat": request["protocol"]["scientific_repeat"], "runtime_digest": digest(request["runtime"]),
        "token_sha256": hashlib.sha256(request["admission"]["token"].encode()).hexdigest(),
        "observation_steps": [row["step"] for row in observations],
        "goal_state_steps": [state["step"] for state in states],
        "completed_updates": completed_steps,
        "started_monotonic": request["admission"]["started_monotonic"],
        "deadline_monotonic": request["admission"]["deadline_monotonic"],
        "attested_monotonic": time.monotonic(),
        "gpu_peak_reserved_mb": reserved_mb, "grade": grade,
        "grade_projection": projection, "files": files})
    guard_source(); admission_guard(request)
    print(json.dumps({"event": "complete", "case": request["task"]["id"], "status": grade["status"]}), flush=True)
    if grade["status"] not in {"PASS", "FAIL"}:
        raise ValueError("canonical evidence did not complete its original protocol")
    return 0 if grade["status"] == "PASS" else 1


def prepare(root, queue_root, output, private, *, task_id, execution_source, mode_hold_packet=None, device=None):
    """Called once inside ROOT's active phase after one complete source freeze."""
    from .canonical_full_atlas_repeat import make_scientific_repeat
    from .queue import host_capacity
    from .sources import compute_profile, runtime_manifest, verify_snapshot
    atomic_json, file_hash, read_json, digest = _utils()
    root, output, private = Path(root).resolve(), Path(output).resolve(), Path(private).resolve()
    if output.exists() or private.exists():
        raise ValueError("a fresh case output/request namespace must not already exist; no reset")
    task = read_json(root / "configs/forge/tasks" / (task_id + ".json"))
    candidate = {"schema_version": 1, "id": "atlas-full-original-common26-ember561",
                 "trainer_family": "atlas", "recipe_preset": "atlas",
                 "recipe_overrides": read_json(root / CONFIG),
                 "initializer": "deterministic_orthogonal", "extensions": {}}
    protocol = read_json(root / PROTOCOL)
    adapter = importlib.import_module(adapter_module(task_id))
    binding = adapter.resolve_binding(root, candidate, task, protocol)
    capacity = host_capacity(); device = _device(task, device)
    gpu = gpu_fit(device) if device != "cpu" else None
    if capacity["cpu_threads"] < 1 or capacity["available_memory_mb"] < HOST_MEMORY_MB:
        raise ValueError("current CPU1/2048-MiB host fit is unavailable")
    runtime = {**runtime_manifest(), "device": device, "cuda_visible_devices": _visibility(task),
               "gpus": int(device != "cpu"), "torch_threads": 1, "deterministic": True, "tf32": False,
               "dtype": "float32", "gpu_memory_limit_mb": GPU_MEMORY_MB if gpu else 0,
               "compute_profile": compute_profile("cuda", gpu["model"], threads=1) if gpu else compute_profile("cpu", threads=1)}
    verify_snapshot(Path(execution_source["snapshot_path"]), execution_source)
    request = {"schema": SCHEMA, "case_id": case_id(task_id), "task": task,
               "candidate": candidate, "protocol": protocol, "binding": binding,
               "source": deepcopy(execution_source), "runtime": runtime,
               "diagnostic": {"original_registered_view": 3, "original_task_slots": 26,
                              "continue_independent_after_numeric_fail": True,
                              "qualification_credit": False, "default_adoption": False,
                              "speed_ranking": False}}
    if task_id == "ring_hold":
        if mode_hold_packet is None:
            raise ValueError("ring group cannot prepare without this campaign's actual passing mode_hold")
        request["prerequisites"] = {"mode_hold": bind_mode_hold_prerequisite(mode_hold_packet)}
        request["grouped_tasks"] = {"ring_hold": deepcopy(task),
                                    "ring_extension": read_json(root / "configs/forge/tasks/ring_extension.json")}
    elif mode_hold_packet is not None:
        raise ValueError("independent tasks cannot be supplied a prior gate packet")
    repeat = make_scientific_repeat(request)
    request["protocol"]["scientific_repeat"] = repeat
    validate_request(request)
    private.mkdir(parents=True)
    frozen_path = private / "frozen-request.json"
    atomic_json(frozen_path, request)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": _visibility(task),
                   "PYTHONPATH": execution_source["snapshot_path"], "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, "-u", "-B", "-m", MODULE, "--preflight", str(frozen_path)],
                          cwd=execution_source["snapshot_path"], env=environment,
                          text=True, capture_output=True, timeout=30)
    (private / "copied-preflight.log").write_text(done.stdout + done.stderr)
    if done.returncode:
        raise ValueError("copied model-free preflight refused: " + done.stderr[-3000:])
    proof = json.loads(done.stdout)
    if proof.get("status") != "PASS" or proof.get("frozen_request_digest") != digest(request):
        raise ValueError("copied-source preflight is not bound to the exact request")
    atomic_json(private / "copied-preflight.json", proof)
    allowance = task["resources"]["timeout_seconds"]
    spec = {"id": case_id(task_id), "representation_card": {"sha256": binding["source_contract_sha256"]},
            "export_grace_seconds": 0, "retries": 0, "frames": 24, "scientific_repeat": repeat,
            "resources": {"host_memory_mb": HOST_MEMORY_MB, "gpu_memory_mb": GPU_MEMORY_MB if gpu else 0}}
    packet = {"schema": SCHEMA, "spec": spec, "spec_sha256": digest(spec), "request": request,
              "protocol": request["protocol"], "scientific_repeat": repeat,
              "execution_source": deepcopy(execution_source),
              "source": {"commit": execution_source["origin_commit"], "digest": execution_source["digest"]},
              "case_definitions": {case_id(task_id): {"task": task, "candidate": candidate,
                                   "protocol": request["protocol"], "binding": binding}},
              "capacity_preflight": {"capacity": capacity, "gpu": gpu,
                                     "required_cpu_threads": 1, "required_host_memory_mb": HOST_MEMORY_MB},
              "runtime_contract": runtime, "lane_runtime": runtime,
              "family_paid_budget_seconds": allowance,
              "rows": [{"id": case_id(task_id), "task_id": task_id, "timeout_seconds": allowance,
                        "allowance_seconds": allowance, "status": "NOT_RUN"}],
              "copied_preflight": proof, "output": str(output), "queue_root": str(Path(queue_root).resolve()),
              "qualification_credit": False, "default_adoption": False, "speed_ranking": False}
    for key, path in (("copied_preflight_pin", private / "copied-preflight.json"),
                      ("frozen_request_pin", frozen_path)):
        packet[key] = {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}
    atomic_json(private / "prepared.json", packet)
    return packet


def frozen_retained_grade(packet, raw_path):
    """Model-free retained grading in the frozen source, not parent's imports."""
    _, file_hash, _, digest = _utils()
    request = packet["request"]
    raw_path = Path(raw_path).resolve()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": _visibility(request),
                   "PYTHONPATH": request["source"]["snapshot_path"], "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-u", "-B", "-m", MODULE, "--grade",
                           packet["frozen_request_pin"]["path"], str(raw_path)],
                          cwd=request["source"]["snapshot_path"], env=environment,
                          text=True, capture_output=True, timeout=30)
    if done.returncode:
        raise ValueError("frozen retained-byte grading refused: " + done.stderr[-3000:])
    proof = json.loads(done.stdout)
    if (proof.get("schema") != "pg_common26_full_atlas_independent_retained_grade_v1"
            or proof.get("source_digest") != request["source"]["digest"]
            or proof.get("frozen_request_digest") != digest(request)
            or proof.get("raw_result_sha256") != file_hash(raw_path)
            or any(type(proof.get(key)) is not int or proof[key] != 0 for key in
                   ("model_constructors", "forwards", "updates", "evaluation_draws"))):
        raise ValueError("frozen independent grade is not bound to these exact retained bytes")
    return proof


@contextmanager
def retained_grade_fences():
    """Static Torch definitions are lawful; physical scientific calls are not."""
    import torch
    import numpy as np
    from unittest.mock import patch
    counts = {"model_constructors": 0, "forwards": 0, "updates": 0, "evaluation_draws": 0}
    def forbidden(kind):
        def refuse(*args, **kwargs):
            counts[kind] += 1
            raise ValueError("retained-byte grading attempted forbidden " + kind)
        return refuse
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module, "__init__", forbidden("model_constructors")))
        stack.enter_context(patch.object(torch.nn.Module, "_call_impl", forbidden("forwards")))
        stack.enter_context(patch.object(torch.optim.Optimizer, "__init__", forbidden("updates")))
        for name in ("rand", "randn", "randint", "rand_like", "randn_like", "randperm", "multinomial", "normal", "bernoulli"):
            stack.enter_context(patch.object(torch, name, forbidden("evaluation_draws")))
        for name in ("normal_", "uniform_", "random_", "bernoulli_", "exponential_", "cauchy_", "log_normal_", "geometric_"):
            stack.enter_context(patch.object(torch.Tensor, name, forbidden("evaluation_draws")))
        for name in ("random", "random_sample", "rand", "randn", "randint", "choice", "normal", "uniform", "shuffle", "permutation"):
            stack.enter_context(patch.object(np.random, name, forbidden("evaluation_draws")))
        yield counts


def certify(packet, target, terminal, token):
    """Independently grade the attested retained bytes, without a model/draw."""
    _, file_hash, read_json, digest = _utils()
    request = packet["request"]; target = Path(target)
    if terminal.get("token") != token or terminal.get("attempt_status") != "completed":
        raise ValueError("matching completed maintained supervisor terminal is required")
    proof = read_json(target / "attestation.json")
    if (proof.get("schema") != "pg_common26_full_atlas_child_attestation_v1"
            or proof.get("frozen_request_digest") != digest(request)
            or proof.get("source_digest") != request["source"]["digest"]
            or proof.get("repeat") != validate_request(request)
            or proof.get("runtime_digest") != digest(request["runtime"])
            or proof.get("token_sha256") != hashlib.sha256(token.encode()).hexdigest()
            or proof.get("started_monotonic") != packet["started_monotonic"]
            or proof.get("deadline_monotonic") != packet["deadline_monotonic"]
            or type(proof.get("attested_monotonic")) not in (int, float)
            or not packet["started_monotonic"] <= proof["attested_monotonic"] < packet["deadline_monotonic"]):
        raise ValueError("child attestation is not this source-bound canonical repeat")
    paid = terminal.get("paid_wall_seconds")
    if (type(paid) not in (int, float) or not math.isfinite(paid) or paid <= 0
            or paid + 1e-6 < proof["attested_monotonic"] - packet["started_monotonic"]):
        raise ValueError("matching durable cost does not cover actual attested child work")
    names = {"INITIALIZATION.json", "MODEL_STARTED.json", "raw-result.json", "goal-states.json",
             "state.pt", "grade.json", "grade-projection.json", "goal.gif", "goal-final.png"}
    if set(proof.get("files", {})) != names:
        raise ValueError("the complete original child artifact manifest is required")
    for name, pin in proof["files"].items():
        if file_hash(target / name) != pin["sha256"] or (target / name).stat().st_size != pin["bytes"]:
            raise ValueError("retained child bytes changed: " + name)
    result = read_json(target / "raw-result.json")
    independent = frozen_retained_grade(packet, target / "raw-result.json")
    grade = independent["grade"]
    if (grade["status"] not in {"PASS", "FAIL"} or grade != proof["grade"]
            or grade != read_json(target / "grade.json")):
        raise ValueError("complete independent retained-byte grade differs")
    projection = independent["grade_projection"]
    if (projection != proof.get("grade_projection")
            or projection != read_json(target / "grade-projection.json")
            or projection["grade"] != grade):
        raise ValueError("independent task/group projection differs from retained child bytes")
    component_owner_health(request, result, proof["completed_updates"])
    expected_exit = 0 if grade["status"] == "PASS" else 1
    if type(terminal.get("child_returncode")) is not int or terminal["child_returncode"] != expected_exit:
        raise ValueError("actual child exit differs from the complete independent grade")
    evidence = result["evidence"]
    observations = evidence.get("observations", evidence.get("dense"))
    clocks = [row["step"] for row in observations]
    states = read_json(target / "goal-states.json")
    goal_clocks = [state["step"] for state in states]
    if (clocks != proof["observation_steps"] or clocks != sorted(set(clocks))
            or goal_clocks != proof["goal_state_steps"] or goal_clocks != sorted(set(goal_clocks))
            or not goal_clocks or any(step not in clocks for step in goal_clocks)
            or type(proof.get("completed_updates")) is not int or proof["completed_updates"] != clocks[-1]):
        raise ValueError("retained actual update/observation/media clocks differ")
    if request["task"]["id"] in IMAGE_TASKS:
        expected_clocks = list(range(25, 601, 25))
        if clocks != expected_clocks or goal_clocks != expected_clocks or proof["completed_updates"] != 600:
            raise ValueError("the complete original image600/24 clock is required")
    if request["task"]["id"] == "ring_hold":
        group = execution_group(request, result)
        if (group != projection["execution_group"] or group["completed_steps"] != proof["completed_updates"]
                or group["prerequisite_proof_sha256"] != request["protocol"]["scientific_repeat"]["prerequisite_proof_sha256"]):
            raise ValueError("ring consumer is not bound to this same physical attempt")
    reserved = proof.get("gpu_peak_reserved_mb")
    if (type(reserved) not in (int, float) or not math.isfinite(reserved)
            or (reserved != 0 if _device(request) == "cpu" else not 0 < reserved <= GPU_MEMORY_MB)):
        raise ValueError("actual admitted GPU memory receipt is invalid")
    initial = read_json(target / "INITIALIZATION.json")
    if (initial.get("repeat") != request["protocol"]["scientific_repeat"] or initial.get("factory_calls") != 1
            or initial.get("historical_checkpoint_loaded") is not False):
        raise ValueError("actual one-time fresh initialization witness is missing")
    start = read_json(target / "MODEL_STARTED.json")
    if (start.get("schema") != "pg_common26_full_atlas_actual_start_v1"
            or start.get("initialization_sha256") != proof["files"]["INITIALIZATION.json"]["sha256"]
            or start.get("frozen_request_digest") != digest(request)
            or start.get("models_constructed") is not True or start.get("completed_updates") != 0
            or start.get("device") != _device(request)
            or start.get("gpus") != int(_device(request) != "cpu")
            or not packet["started_monotonic"] <= start["initialized_monotonic"] < packet["deadline_monotonic"]):
        raise ValueError("the actual model did not start inside this admitted deadline")
    return {"grade": grade, "grade_projection": projection, "actual_model_started": start,
            "attestation_sha256": file_hash(target / "attestation.json"),
            "result_sha256": proof["files"]["raw-result.json"]["sha256"],
            "independent_retained_grade": independent,
            "media": "goal.gif", "full_protocol_complete": True,
            "qualification_credit": False, "default_adoption": False, "speed_ranking": False}


def run(packet, output, *, parent, budget, predecessors, campaign_rows, dependency_outcomes):
    """One maintained physical attempt; complete FAIL permits independent next cases."""
    from .policy_execution import PolicyCoordinator
    from .sources import verify_snapshot
    atomic_json, file_hash, read_json, digest = _utils()
    output = Path(output).resolve(); request = packet["request"]
    for field, expected in (("frozen_request_pin", request), ("copied_preflight_pin", packet["copied_preflight"])):
        pin = packet[field]; path = Path(pin["path"])
        if file_hash(path) != pin["sha256"] or path.stat().st_size != pin["bytes"] or read_json(path) != expected:
            raise ValueError("persisted copied-source prerequisite changed: " + field)
    proof = packet["copied_preflight"]
    if (proof.get("schema") != "pg_common26_full_atlas_copied_preflight_v1" or proof.get("status") != "PASS"
            or proof.get("frozen_request_digest") != digest(request)
            or proof.get("source_digest") != request["source"]["digest"]
            or proof.get("repeat") != validate_request(request)
            or proof.get("source_contract_sha256") != request["binding"]["source_contract_sha256"]
            or any(type(proof.get(key)) is not int or proof[key] != 0 for key in
                   ("model_constructors", "forwards", "updates", "evaluation_draws"))):
        raise ValueError("exact zero-model copied-source preflight is missing before reservation")
    dependency = dependency_status(request["task"], dependency_outcomes, request=request)
    if dependency["status"] != "READY":
        return {"status": "BLOCKED", "blocker_kind": "dependency", "source_bound_dependency": dependency,
                "models": 0, "case_reservation_seconds": 0}
    budget.require_next_case_fit(budget._maintained, parent.snapshot(), predecessors=predecessors,
                                 rows=campaign_rows, next_task_id=request["task"]["id"])
    verify_snapshot(Path(request["source"]["snapshot_path"]), request["source"])
    device = _device(request)
    if device != "cpu":
        gpu_fit(device)
    coordinator = PolicyCoordinator(packet["queue_root"])
    key, canonical = coordinator.register(packet, output, "atlas", packet["lane_runtime"])
    if canonical.resolve() != output:
        raise ValueError("fresh diagnostic cannot attach to an older canonical study")
    row = packet["rows"][0]
    trial = {"family": "atlas", "recipe_overrides": request["candidate"]["recipe_overrides"]}
    attempt = coordinator.attempt_key(packet, trial, row)
    if coordinator.retained(attempt) is not None:
        raise ValueError("recognized fresh diagnostic already has an attempt; no reuse or retry")
    allowance = row["allowance_seconds"]
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = _visibility(request)
    try:
        with coordinator.study_lease(key) as study:
            if study is None:
                raise ValueError("this canonical diagnostic study lease is busy")
            with coordinator.admit(attempt, packet, row, device) as (admitted, lease):
                if admitted["status"] != "running" or lease is None:
                    raise ValueError("physical admission refused: " + admitted.get("reason", admitted["status"]))
                packet.update(started_monotonic=admitted["started_monotonic"], deadline_monotonic=admitted["deadline_monotonic"])
                target = output / request["task"]["id"]
                terminal_path = Path(admitted["lease_path"]).parent / "supervisor-terminal.json"
                certificate = None
                try:
                    target.mkdir()
                    admitted_request = deepcopy(request); path = target / "request.json"
                    command = [sys.executable, "-u", "-B", "-m", MODULE, "--child", str(path)]
                    admitted_request.update(target=str(target), command=command, admission={
                        "token": admitted["token"], "started_monotonic": admitted["started_monotonic"],
                        "deadline_monotonic": admitted["deadline_monotonic"],
                        "lease_fds": [study.fileno(), lease.fileno()], "lease_paths": [str(study.name), str(lease.name)]})
                    atomic_json(path, admitted_request)
                    row.update(status="RUNNING", attempt_key=attempt)
                    atomic_json(output / "study.json", packet)
                    print(json.dumps({"event": "launch", "case": request["task"]["id"],
                                      "allowance_seconds": allowance, "source_digest": request["source"]["digest"]}), flush=True)
                    with parent.pause():
                        coordinator.launch(command, packet, target / "run.log", (study, lease), allowance)
                    terminal = read_json(terminal_path)
                    certificate = certify(packet, target, terminal, admitted["token"])
                    status, reason = certificate["grade"]["status"], certificate["grade"].get("reason")
                except Exception as error:
                    terminal = read_json(terminal_path) if terminal_path.exists() else None
                    status = "INCOMPLETE" if isinstance(error, subprocess.TimeoutExpired) else "INVALID"
                    reason = type(error).__name__ + ": " + str(error)
                matched = terminal if terminal and terminal.get("token") == admitted["token"] else None
                accepted = (certificate is not None and status in {"PASS", "FAIL"}
                            and matched is not None and matched["paid_wall_seconds"] <= allowance)
                cost = budget.case_cost(budget._maintained, matched, allowance_seconds=allowance,
                                        expected_token=admitted["token"], certified=accepted)
                if cost["overrun_seconds"] > 0:
                    status, reason, accepted = "INCOMPLETE", "actual case exceeded its unchanged allowance", False
                    if certificate is not None:
                        certificate["full_protocol_complete"] = False
                result = {"status": status, "reason": reason, "certificate": certificate,
                          "certified": accepted, "completed_terminal": bool(matched and matched.get("attempt_status") == "completed"),
                          "terminal_status": matched.get("attempt_status") if matched else None,
                          "allowance_seconds": allowance, **cost}
                coordinator.complete(attempt, result)
                row.update(result)
                packet["progression"] = {"continue_independent": accepted, "halt_required": not accepted,
                                         "failed_result_preserved": status == "FAIL", "qualification_credit": False,
                                         "default_adoption": False, "speed_ranking": False}
                atomic_json(output / "study.json", packet)
                return packet
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def verify_prepared_packet(packet):
    """Paid exact copied proof validation before any wave reservation."""
    _, file_hash, read_json, digest = _utils()
    request = packet["request"]
    for field, expected in (("frozen_request_pin", request), ("copied_preflight_pin", packet["copied_preflight"])):
        pin = packet[field]; path = Path(pin["path"])
        if file_hash(path) != pin["sha256"] or path.stat().st_size != pin["bytes"] or read_json(path) != expected:
            raise ValueError("persisted copied-source prerequisite changed: " + field)
    proof = packet["copied_preflight"]
    if (proof.get("schema") != "pg_common26_full_atlas_copied_preflight_v1" or proof.get("status") != "PASS"
            or proof.get("frozen_request_digest") != digest(request)
            or proof.get("source_digest") != request["source"]["digest"]
            or proof.get("repeat") != validate_request(request)
            or proof.get("source_contract_sha256") != request["binding"]["source_contract_sha256"]
            or any(type(proof.get(key)) is not int or proof[key] != 0 for key in
                   ("model_constructors", "forwards", "updates", "evaluation_draws"))):
        raise ValueError("exact zero-model copied-source preflight is missing before reservation")
    validate_request(request)


def wave_layout(packets):
    """Pure runtime/resource joins; at most one owner per physical card."""
    if not isinstance(packets, list) or not 1 <= len(packets) <= 2:
        raise ValueError("one or two fixed independent cases are required")
    requests = [packet["request"] for packet in packets]
    tasks = [request["task"]["id"] for request in requests]
    for task in tasks:
        adapter_module(task)
    devices = [_device(request) for request in requests]
    if len(set(tasks)) != len(tasks) or len(set(devices)) != len(devices):
        raise ValueError("a wave cannot retry a task or overlap a physical card")
    if len(packets) > 1 and ("cpu" in devices or set(devices) != {"cuda:0", "cuda:1"}):
        raise ValueError("AE is CPU-only and solo; a parallel wave uses both distinct cards")
    if len({_visibility(request) for request in requests}) != 1:
        raise ValueError("parallel launch must have one immutable shared visibility")
    if any(request["source"] != requests[0]["source"]
           or request["candidate"] != requests[0]["candidate"]
           or _base_protocol(request["protocol"]) != _base_protocol(requests[0]["protocol"])
           for request in requests[1:]):
        raise ValueError("a wave cannot blend source, full config or scientific protocol")
    return {"task_ids": tasks, "devices": devices, "visibility": _visibility(requests[0])}


def reserve_admitted_row(row, *, budget, attempt, allowance, token):
    """Pure full-reserve metadata; persisted immediately after actual admission."""
    cost = budget.case_cost(budget._maintained, None, allowance_seconds=allowance,
                            expected_token=token, certified=False)
    row.update(cost)
    row.update(status="RUNNING", attempt_key=attempt)
    return row


def finish_staged_job(job, *, budget, launch_error=None):
    """Paid original retained certification and one durable physical cost."""
    atomic_json, _, read_json, _ = _utils()
    packet, admitted = job["packet"], job["admitted"]
    row, allowance = packet["rows"][0], job["allowance"]
    certificate = None
    terminal_path = job["terminal_path"]
    terminal = read_json(terminal_path) if terminal_path.exists() else None
    try:
        if launch_error is not None:
            raise launch_error
        certificate = certify(packet, job["target"], terminal, admitted["token"])
        status = certificate["grade"]["status"]
        reason = certificate["grade"].get("reason")
    except BaseException as error:
        status = "INCOMPLETE" if isinstance(error, subprocess.TimeoutExpired) else "INVALID"
        reason = type(error).__name__ + ": " + str(error)
    matched = terminal if terminal and terminal.get("token") == admitted["token"] else None
    accepted = (certificate is not None and status in {"PASS", "FAIL"}
                and matched is not None and matched["paid_wall_seconds"] <= allowance)
    cost = budget.case_cost(budget._maintained, matched, allowance_seconds=allowance,
                            expected_token=admitted["token"], certified=accepted)
    if cost["overrun_seconds"] > 0:
        status, reason, accepted = "INCOMPLETE", "actual case exceeded its unchanged allowance", False
        if certificate is not None:
            certificate["full_protocol_complete"] = False
    result = {"status": status, "reason": reason, "certificate": certificate,
              "certified": accepted, "completed_terminal": bool(matched and matched.get("attempt_status") == "completed"),
              "terminal_status": matched.get("attempt_status") if matched else None,
              "allowance_seconds": allowance, **cost}
    job["coordinator"].complete(job["attempt"], result)
    row.update(result)
    packet["progression"] = {"continue_independent": accepted, "halt_required": not accepted,
                             "failed_result_preserved": status == "FAIL", "qualification_credit": False,
                             "default_adoption": False, "speed_ranking": False}
    atomic_json(job["output"] / "study.json", packet)
    return packet


def started_job_receipt(job):
    """Paid durable initial owner + supervisor child join, before completion."""
    _, file_hash, read_json, digest = _utils()
    path = job["target"] / "MODEL_STARTED.json"
    child_path = job["terminal_path"].parent / "child.json"
    if not path.exists() or not child_path.exists():
        return None
    start, child = read_json(path), read_json(child_path)
    request, admitted = job["packet"]["request"], job["admitted"]
    if (start.get("schema") != "pg_common26_full_atlas_actual_start_v1"
            or start.get("models_constructed") is not True or start.get("completed_updates") != 0
            or start.get("pid") != child.get("pid") or child.get("token") != admitted["token"]
            or start.get("device") != _device(request)
            or start.get("frozen_request_digest") != digest(request)
            or start.get("initialization_sha256") != file_hash(job["target"] / "INITIALIZATION.json")
            or not admitted["started_monotonic"] <= start["initialized_monotonic"] < admitted["deadline_monotonic"]):
        raise ValueError("actual model start is not this owned admitted child")
    return {"case": request["task"]["id"], "device": _device(request), "pid": child["pid"],
            "process_identity": child["process_identity"], "source_digest": request["source"]["digest"],
            "started_monotonic": start["initialized_monotonic"],
            "model_started_sha256": file_hash(path), "qualification_credit": False}


def run_wave(packets, *, parent, budget, predecessors, campaign_rows,
             dependency_outcomes, on_start=None):
    """One admitted pair; setup/certification paid, only parallel launch waits paused."""
    from .policy_execution import PolicyCoordinator
    from .sources import verify_snapshot
    atomic_json, _, _, _ = _utils()
    layout = wave_layout(packets)
    readiness = {}
    for packet in packets:
        verify_prepared_packet(packet)
        request = packet["request"]
        readiness[request["task"]["id"]] = dependency_status(request["task"], dependency_outcomes, request=request)
    budget.require_wave_fit(budget._maintained, parent.snapshot(), predecessors=predecessors,
                            rows=campaign_rows, wave_task_ids=layout["task_ids"], readiness=readiness)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = layout["visibility"]
    jobs = []
    try:
        with ExitStack() as stack:
            try:
                for packet in packets:
                    request = packet["request"]; output = Path(packet["output"]).resolve()
                    verify_snapshot(Path(request["source"]["snapshot_path"]), request["source"])
                    device = _device(request)
                    if device != "cpu":
                        gpu_fit(device)
                    coordinator = PolicyCoordinator(packet["queue_root"])
                    key, canonical = coordinator.register(packet, output, "atlas", packet["lane_runtime"])
                    if canonical.resolve() != output:
                        raise ValueError("fresh diagnostic cannot attach to another canonical study")
                    row = packet["rows"][0]
                    trial = {"family": "atlas", "recipe_overrides": request["candidate"]["recipe_overrides"]}
                    attempt = coordinator.attempt_key(packet, trial, row)
                    if coordinator.retained(attempt) is not None:
                        raise ValueError("no overlap, reuse or retry of this fixed case")
                    study = stack.enter_context(coordinator.study_lease(key))
                    if study is None:
                        raise ValueError("fresh study lease is unavailable")
                    admitted, lease = stack.enter_context(coordinator.admit(attempt, packet, row, device))
                    if admitted["status"] != "running" or lease is None:
                        raise ValueError("physical admission refused: " + admitted.get("reason", admitted["status"]))
                    packet.update(started_monotonic=admitted["started_monotonic"], deadline_monotonic=admitted["deadline_monotonic"])
                    target = output / request["task"]["id"]
                    job = {"packet": packet, "admitted": admitted, "coordinator": coordinator, "attempt": attempt,
                           "allowance": row["allowance_seconds"], "target": target, "output": output,
                           "terminal_path": Path(admitted["lease_path"]).parent / "supervisor-terminal.json",
                           "leases": (study, lease)}
                    jobs.append(job)  # An admitted attempt is never erased by a later setup error.
                    reserve_admitted_row(row, budget=budget, attempt=attempt,
                                         allowance=job["allowance"], token=admitted["token"])
                    # Conservative full allowance survives a parent interruption
                    # before a terminal exists. Completion replaces it exactly once.
                    atomic_json(output / "study.json", packet)
                    target.mkdir()
                    path = target / "request.json"
                    command = [sys.executable, "-u", "-B", "-m", MODULE, "--child", str(path)]
                    job["command"] = command
                    admitted_request = deepcopy(request)
                    admitted_request.update(target=str(target), command=command, admission={
                        "token": admitted["token"], "started_monotonic": admitted["started_monotonic"],
                        "deadline_monotonic": admitted["deadline_monotonic"],
                        "lease_fds": [study.fileno(), lease.fileno()], "lease_paths": [str(study.name), str(lease.name)]})
                    atomic_json(path, admitted_request)
                    row.update(status="RUNNING", attempt_key=attempt)
                    atomic_json(output / "study.json", packet)
            except BaseException as error:
                failed = [finish_staged_job(job, budget=budget, launch_error=error) for job in jobs]
                if failed:
                    return failed
                raise
            pool = None; futures = {}; errors = {}; seen = set(); orchestration_error = None
            try:
                pool = ThreadPoolExecutor(max_workers=len(jobs))
                for index, job in enumerate(jobs):
                    future = pool.submit(job["coordinator"].launch, job["command"], job["packet"],
                                         job["target"] / "run.log", job["leases"], job["allowance"])
                    futures[future] = index
                pending = set(futures)
                while pending:
                    with parent.pause():
                        _, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    if on_start is not None:
                        for index, job in enumerate(jobs):
                            if index in seen:
                                continue
                            receipt = started_job_receipt(job)
                            if receipt is not None:
                                on_start(receipt)
                                seen.add(index)
            except BaseException as error:
                orchestration_error = error
            finally:
                if pool is not None:
                    # Drain already admitted, bounded maintained launchers before
                    # certifying. No callback failure erases their durable cost.
                    with parent.pause():
                        pool.shutdown(wait=True)
            for future, index in futures.items():
                try:
                    future.result()
                except BaseException as error:
                    errors[index] = error
            submitted = set(futures.values())
            if orchestration_error is not None:
                for index in set(range(len(jobs))) - submitted:
                    errors[index] = orchestration_error
            finished = [finish_staged_job(job, budget=budget, launch_error=errors.get(index))
                        for index, job in enumerate(jobs)]
            if orchestration_error is not None:
                for job, packet in zip(jobs, finished, strict=True):
                    # Retain numerical grade/certificate and measured completed
                    # cost, but the parent failure prevents another admission.
                    packet["parent_orchestration_error"] = type(orchestration_error).__name__ + ": " + str(orchestration_error)
                    packet["rows"][0]["parent_orchestration_error"] = packet["parent_orchestration_error"]
                    packet["rows"][0]["scientific_certificate_preserved"] = packet["rows"][0].get("certificate") is not None
                    packet["rows"][0]["certified"] = False
                    packet["progression"].update(continue_independent=False, halt_required=True)
                    atomic_json(job["output"] / "study.json", packet)
            return finished

    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def main():
    _, file_hash, read_json, digest = _utils()
    if len(sys.argv) == 4 and sys.argv[1] == "--grade":
        request = read_json(sys.argv[2]); root = Path(__file__).resolve().parents[2]
        source_guard(request, root)
        raw_path = Path(sys.argv[3]); result = read_json(raw_path)
        with retained_grade_fences() as counts:
            from .views import grade_result
            projection = grade_projection(request, result, grade_result)
            grade = projection["grade"]
        source_guard(request, root)
        print(json.dumps({"schema": "pg_common26_full_atlas_independent_retained_grade_v1",
                          "source_digest": request["source"]["digest"], "frozen_request_digest": digest(request),
                          "raw_result_sha256": file_hash(raw_path), "grade": grade,
                          "grade_projection": projection,
                          "scientific_call_fences": "physical_Module_optimizer_and_RNG_calls_refused",
                          **counts}, sort_keys=True))
        return 0
    if len(sys.argv) != 3 or sys.argv[1] not in {"--preflight", "--child"}:
        raise SystemExit("Use ROOT's paid parent; only exact copied preflight or child is supported")
    if sys.argv[1] == "--preflight":
        print(json.dumps(preflight(read_json(sys.argv[2]), Path(__file__).resolve().parents[2]), sort_keys=True))
        return 0
    return child(sys.argv[2])


if __name__ == "__main__":
    raise SystemExit(main())
