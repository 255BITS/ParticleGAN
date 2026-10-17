"""Recognized common-task intents and admitted one-factory initialization proof.

Metadata functions import no scientific package, read no files and create no
owners. Only construct() inspects the actual owner in the admitted child.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import sys

PACKET_SCHEMA = "pg_common26_full_atlas_diagnostic_case_v1"
REPEAT_SCHEMA = "forge_canonical_full_atlas_fresh_repeat_v1"
MODULE = "experiments/forge/canonical_full_atlas_repeat.py"
CASE_PREFIX = "canonical-common26-full-atlas-"
CONFIG_FILE_SHA256 = "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"
CONFIG_DIGEST = "6ce21adbd06cb20a1e9725876cf19025f7bb507a7c7ffeba6bf95b2c1f5fdfec"
PROTOCOL_FILE_SHA256 = "3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803"
PROTOCOL_DIGEST = "7b975bab8008c880a88e04d9e9bbdf4d1ab549d46d1d894b9e42369fdc0931b2"
_CONSTRUCTIONS = []


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid " + label)
    return value


def _base(request):
    return {key: deepcopy(value) for key, value in request.items()
            if key not in {"admission", "target", "command"}}


def case_id(task_id):
    return CASE_PREFIX + task_id + "-ember561-v1"


def base_protocol(value):
    protocol = deepcopy(value)
    if not isinstance(protocol, dict):
        raise ValueError("the exact common screening protocol is required")
    if "scientific_repeat" in protocol and not isinstance(protocol.pop("scientific_repeat"), dict):
        raise ValueError("malformed scientific repeat")
    if digest(protocol) != PROTOCOL_DIGEST:
        raise ValueError("changed screening protocol, seed or RNG law")
    return protocol


def make_scientific_repeat(request):
    """Source-bound intended repeat; it is never proof of initialization."""
    request = _base(request)
    tid = request.get("task", {}).get("id")
    if tid not in ROSTER:
        raise ValueError("unsupported fresh owner or independent ring-extension replay")
    item = ROSTER[tid]
    fields = {"schema", "case_id", "task", "candidate", "protocol", "binding", "source", "runtime", "diagnostic"}
    if tid == "ring_hold":
        fields |= {"prerequisites", "grouped_tasks"}
    if (set(request) != fields or request["schema"] != PACKET_SCHEMA
            or request["case_id"] != case_id(tid) or digest(request["task"]) != item["task_digest"]):
        raise ValueError("the exact sole canonical task and recognized case are required")
    protocol = base_protocol(request["protocol"])
    candidate = request["candidate"]
    if (set(candidate) != {"schema_version", "id", "trainer_family", "recipe_preset",
                          "recipe_overrides", "initializer", "extensions"}
            or type(candidate["schema_version"]) is not int or candidate["schema_version"] != 1
            or candidate["id"] != "atlas-full-original-common26-ember561"
            or candidate["trainer_family"] != "atlas" or candidate["recipe_preset"] != "atlas"
            or digest(candidate["recipe_overrides"]) != CONFIG_DIGEST
            or candidate["initializer"] != "deterministic_orthogonal" or candidate["extensions"] != {}):
        raise ValueError("only the complete frozen original Atlas declaration is recognized")
    binding = request["binding"]
    if (digest(binding) != item["binding_sha256"] or binding.get("schema") != item["binding_schema"]
            or digest(binding["recipe"]) != item["recipe_sha256"]
            or digest(binding["source_contract"]) != item["source_contract_sha256"]
            or binding["source_contract_sha256"] != item["source_contract_sha256"]):
        raise ValueError("changed source-derived owner, complete Recipe, sampling or gates")
    runtime = request["runtime"]
    device = "cpu" if item["kind"] == "ae" else (runtime.get("device") if isinstance(runtime, dict) else None)
    if item["kind"] != "ae" and (type(device) is not str or device not in {"cuda:0", "cuda:1"}):
        raise ValueError("only physical CUDA0 or CUDA1 is declared for this campaign")
    expected_runtime = dict(device=device, cuda_visible_devices="" if device == "cpu" else "0,1",
        gpus=int(device != "cpu"), torch_threads=1, deterministic=True, tf32=False,
        dtype="float32", gpu_memory_limit_mb=0 if device == "cpu" else 2048)
    if (not isinstance(runtime, dict) or not isinstance(runtime.get("compute_profile"), dict)
            or any(type(runtime.get(key)) is not type(value) or runtime.get(key) != value
                   for key, value in expected_runtime.items())):
        raise ValueError("declared CPU1, device, memory, dtype and deterministic runtime required")
    if request["diagnostic"] != dict(original_registered_view=3, original_task_slots=26,
            continue_independent_after_numeric_fail=True, qualification_credit=False,
            default_adoption=False, speed_ranking=False):
        raise ValueError("fresh diagnostic cannot acquire qualification or default credit")
    source = request["source"]
    origin = source.get("origin_commit")
    if (not isinstance(origin, str) or len(origin) != 40
            or any(c not in "0123456789abcdef" for c in origin)):
        raise ValueError("exact source origin is required")
    files = source.get("files")
    if (not isinstance(files, dict) or not files
            or any(not isinstance(member, str) or member.startswith("/") or "\\" in member
                   or any(part in {"", ".", ".."} for part in member.split("/")) for member in files)
            or any(_sha(value, "source member") != value for value in files.values())
            or source.get("digest") != digest(files)):
        raise ValueError("complete source file map and its exact digest are required")
    snapshot = source.get("snapshot_path")
    if not isinstance(snapshot, str) or not snapshot.startswith("/"):
        raise ValueError("source snapshot identity is missing")
    adapter = "experiments/forge/canonical_" + item["kind"] + "_adapter.py"
    for member in (MODULE, adapter, "experiments/forge/canonical_full_atlas_case.py",
                   "experiments/forge/policy_execution.py"):
        _sha(files.get(member), "owned source member " + member)
    if files[adapter] != item["adapter_sha256"]:
        raise ValueError("adapter is not its exact declared source member")
    pins = binding["source_contract"]["files"]
    for member, pin in pins.items():
        expected = pin["sha256"]
        if files.get(member) != expected:
            raise ValueError("canonical source dependency changed: " + member)
    if (files.get("configs/100gaussians/atlas.json") != CONFIG_FILE_SHA256
            or files.get("configs/forge/protocols/screening.json") != PROTOCOL_FILE_SHA256
            or files.get("configs/forge/tasks/" + tid + ".json") != item["task_file_sha256"]):
        raise ValueError("canonical source declarations changed")
    result = dict(schema=REPEAT_SCHEMA,
        repeat_id="canonical-common26-full-atlas-" + tid + "-ember561-fresh-init-seed0-v1",
        case_id=case_id(tid), task_id=tid, authority_sequence=561,
        seed=protocol["seed"], rng_version=protocol["rng"]["derivation"],
        task_digest=item["task_digest"], protocol_digest=PROTOCOL_DIGEST,
        candidate_digest=digest(candidate), binding_digest=item["binding_sha256"],
        config_sha256=CONFIG_FILE_SHA256, effective_recipe_digest=item["recipe_sha256"],
        source_contract_sha256=item["source_contract_sha256"], source_origin_commit=origin,
        source_digest=source["digest"], source_manifest_digest=digest(source),
        runtime_digest=digest(runtime), initialization="new_owner_factory",
        execution_updates=item["updates"], recipe_schedule_horizon=None,
        observation_contract_sha256=item["observation_contract_sha256"],
        accepted_numeric_credit=False)
    if tid == "ring_hold":
        group = binding["source_contract"]["grouped_execution"]
        if (digest(request["grouped_tasks"]) != digest(group["task_definitions"])
                or set(request["prerequisites"]) != {"mode_hold"}
                or not isinstance(request["prerequisites"]["mode_hold"], dict)
                or not request["prerequisites"]["mode_hold"]):
            raise ValueError("uninterrupted group and current bound mode-hold proof are required")
        # The source-bound controller validates terminal/artifact bytes and
        # independently recertifies the prerequisite before construction.
        result.update(prerequisite_proof_sha256=digest(request["prerequisites"]["mode_hold"]),
                      grouped_tasks_sha256=digest(request["grouped_tasks"]))
    return result


def validate_scientific_repeat(request):
    expected = make_scientific_repeat(request)
    if digest(request["protocol"].get("scientific_repeat")) != digest(expected):
        raise ValueError("absent or altered recognized fresh repeat intent")
    return deepcopy(expected)


def coordinator_repeat(packet, trial, row):
    """Bind only the prepared sole row; history or a new campaign is no repeat."""
    if packet.get("schema") != PACKET_SCHEMA:
        raise ValueError("foreign common-task repeat envelope")
    request = packet["request"]
    repeat = validate_scientific_repeat(request)
    tid = repeat["task_id"]
    cap = ROSTER[tid]["timeout_seconds"]
    expected_row = dict(id=case_id(tid), task_id=tid, timeout_seconds=cap,
                        allowance_seconds=cap, status="NOT_RUN")
    expected_case = {key: deepcopy(request[key]) for key in ("task", "candidate", "protocol", "binding")}
    spec = dict(id=case_id(tid), representation_card=dict(sha256=repeat["source_contract_sha256"]),
        export_grace_seconds=0, retries=0, frames=24, scientific_repeat=repeat,
        resources=dict(host_memory_mb=2048, gpu_memory_mb=0 if tid == "ae_gan_hold" else 2048))
    if (digest(row) != digest(expected_row) or digest(packet.get("rows")) != digest([expected_row])
            or digest(packet.get("case_definitions")) != digest({case_id(tid): expected_case})
            or digest(packet.get("spec")) != digest(spec) or packet.get("spec_sha256") != digest(spec)
            or digest(packet.get("protocol")) != digest(request["protocol"])
            or digest(packet.get("scientific_repeat")) != digest(repeat)
            or digest(packet.get("execution_source")) != digest(request["source"])
            or packet.get("source") != dict(commit=repeat["source_origin_commit"], digest=repeat["source_digest"])
            or digest(packet.get("lane_runtime")) != digest(request["runtime"])
            or digest(packet.get("runtime_contract")) != digest(request["runtime"])
            or type(packet.get("family_paid_budget_seconds")) is not int
            or packet["family_paid_budget_seconds"] != cap
            or trial.get("family") != "atlas"
            or digest(trial.get("recipe_overrides")) != digest(request["candidate"]["recipe_overrides"])
            or any(packet.get(key) is not False for key in ("qualification_credit", "default_adoption", "speed_ranking"))):
        raise ValueError("prepared repeat cannot change its identity or carry historical credit")
    return repeat


def _actual(value, module, name):
    loaded = sys.modules.get(module)
    expected = None if loaded is None else getattr(loaded, name, None)
    if not isinstance(expected, type) or type(value) is not expected:
        raise ValueError("actual canonical " + module + "." + name + " is required")


def _zero(value):
    return type(value) is int and value == 0


class FreshRepeatGuard:
    """Require the admitted factory, actual owner and zero public clocks once."""
    def __init__(self, request, *, source_guard, admission_guard):
        if not callable(source_guard) or not callable(admission_guard):
            raise TypeError("admitted source and lease guards are required")
        self.repeat = validate_scientific_repeat(request)
        self._request = deepcopy(request)
        self._source_guard, self._admission_guard = source_guard, admission_guard
        self._phase, self._owner, self.initialization_witness = "pending", None, None

    def construct(self, factory, initial_reader):
        if self._phase != "pending":
            raise RuntimeError("one fresh repeat permits one factory and no retry")
        if not callable(factory) or not callable(initial_reader):
            raise TypeError("source-bound factory and initial owner reader are required")
        self._phase = "constructing"
        try:
            self._source_guard(); self._admission_guard()
            owner = factory()
            kind = ROSTER[self.repeat["task_id"]]["kind"]
            module = "experiments.forge.canonical_" + kind + "_adapter"
            _actual(owner, module, {"image": "ImageOwner", "mog": "MogOwner", "ae": "AEOwner"}[kind])
            loaded = sys.modules[module]
            if initial_reader is not getattr(loaded, "owner_initial_receipt", None):
                raise ValueError("only the canonical source-bound owner reader is accepted")
            token = owner.construction_id
            if (type(token) is not object or owner.restored is not False
                    or any(owner is old or token is old_token for old, old_token in _CONSTRUCTIONS)):
                raise ValueError("cached or restored owner cannot witness new initialization")
            _CONSTRUCTIONS.append((owner, token))
            for value, mod, name in ((owner.policy, "particlegan.policy", "UpdatePolicy"),
                (owner.recipe, "particlegan.recipes", "Recipe"),
                (owner.opt_g, "particlegan.k3p", "K3PGeneratorAdam"),
                (owner.opt_d, "particlegan.ka2", "KA2CriticAdam"),
                (owner.streams, "experiments.forge.rng", "NamedStreams"),
                (owner.prior.z, "torch.nn.parameter", "Parameter")):
                _actual(value, mod, name)
            _actual(owner.prior, "particlegan.particle_prior", "ParticlePrior" if kind == "image" else "MoGParticlePrior")
            if kind == "image":
                _actual(owner.trainer, "particlegan.training", "GANTrainer")
                _actual(owner.generator, "benchmarks.transfer_suite.image_tasks", "Generator")
                _actual(owner.critic, "benchmarks.transfer_suite.image_tasks", "Discriminator")
            elif kind == "mog":
                if type(owner.trainer) is not owner.trainer_class:
                    raise ValueError("declared supplied-prior trainer type required")
                _actual(owner.generator, "lib.toy_models", "SimpleMLPGenerator")
                _actual(owner.critic, "lib.toy_models", "SimpleMLPDiscriminator")
            else:
                for model in (owner.generator, owner.encoder, owner.critic):
                    _actual(model, "benchmarks.locked_shared.hosts.ae_gan_hold", "MLP")
            policy = owner.policy
            recipe = self._request["binding"]["recipe"]
            if (digest(asdict(owner.recipe)) != self.repeat["effective_recipe_digest"]
                    or digest(owner.binding) != self.repeat["binding_digest"]
                    or policy.recipe is not owner.recipe or policy.G is not owner.generator
                    or policy.D is not owner.critic or policy.prior is not owner.prior
                    or policy.table is not owner.prior.z or policy.opt_g is not owner.opt_g
                    or policy.opt_d is not owner.opt_d or policy.table_optimizer is not owner.opt_g
                    or not _zero(policy.completed_steps) or policy._phase != "ready"
                    or owner.opt_g.state or owner.opt_d.state
                    or type(owner.streams.seed) is not int or owner.streams.seed != 0
                    or owner.streams.version != "forge-rng-v1"
                    or list(owner.prior.z.shape) != [recipe["num_particles"], recipe["z_dim"]]
                    or owner.prior.z.requires_grad is not True
                    or str(owner.prior.z.dtype) != "torch.float32"
                    or str(owner.prior.z.device) != self._request["runtime"]["device"]
                    or policy._fast is not None or policy.averaged_table is owner.prior.z
                    or policy.ema_G is owner.generator or policy.ema_prior is owner.prior):
                raise ValueError("actual owner differs from the frozen zero-update public state")
            required_owners = ("controller", "lr_settle", "birth_death", "row_evidence",
                               "reopen_guard", "surprise", "log_output_sigma", "_feature_selection")
            hook_names = {"begin_step", "after_critic_step", "after_generator_backward",
                          "after_generator_step", "finish_step"}
            if kind == "ae":
                hook_names |= {"before_critic_backward", "before_generator_backward"}
            if (any(getattr(policy, name, None) is None for name in required_owners)
                    or not _zero(policy.controller.updates) or not _zero(owner.opt_d.record.observed_steps)
                    or owner.opt_g.latent_damping is None or not _zero(owner.opt_g.latent_damping.total)
                    or owner.opt_g.latent_damping.started is not False
                    or set(owner.calls) != hook_names
                    or any(not _zero(value) for value in owner.calls.values())):
                raise ValueError("every requested policy owner and zero public lifecycle clock is required")
            _actual(policy.surprise, "particlegan.continuous", "OptimizerSurprise")
            if (not _zero(policy.surprise.fires) or not _zero(policy.surprise.streak)
                    or policy.surprise.fast or policy.surprise.slow or policy.surprise.pending):
                raise ValueError("actual optimizer-surprise owner has prior update evidence")
            if kind != "ae":
                trainer = owner.trainer
                if (not _zero(trainer.completed_steps) or trainer.G is not owner.generator
                        or trainer.D is not owner.critic or trainer.prior is not owner.prior
                        or trainer.policy is not policy or trainer.opt_g is not owner.opt_g
                        or trainer.opt_d is not owner.opt_d
                        or trainer.max_steps != ROSTER[self.repeat["task_id"]]["updates"]):
                    raise ValueError("actual trainer and external limit are not bound")
                if kind == "mog" and owner.context.policy_task is not None:
                    raise ValueError("selected-cloud context is not the canonical ordinary task")
                _actual(owner.lifecycle_audit, "experiments.forge.policy_adapters", "PolicyLifecycleAudit")
                if (owner.lifecycle_audit.calls is not owner.calls or owner.lifecycle_audit.pending
                        or not _zero(owner.lifecycle_audit.order_errors)):
                    raise ValueError("new public lifecycle audit must have no prior hook or error")
            else:
                if (policy.encoder is not owner.encoder or policy.ema_encoder is None
                        or policy.roles != [["generator", "encoder", "table", "noise"], ["critic"]]):
                    raise ValueError("actual separately owned auxiliary encoder is required")
            empty_fields = ("observation_purity", "retained_goal_states") if kind == "mog" else ("observations", "purity", "retained")
            if kind == "ae":
                empty_fields += ("losses",)
            if any(getattr(owner, field) for field in empty_fields):
                raise ValueError("initial owner cannot carry old observations, media or losses")
            modules = [owner.generator, owner.critic, owner.prior] + ([owner.encoder] if kind == "ae" else [])
            expected_parameters = {id(p) for model in modules for p in model.parameters() if p.requires_grad}
            expected_parameters.add(id(policy.log_output_sigma))
            parameters = [p for optimizer in (owner.opt_g, owner.opt_d)
                          for group in optimizer.param_groups for p in group["params"]]
            if (len(parameters) != len({id(p) for p in parameters}) or {id(p) for p in parameters} != expected_parameters
                    or any(str(p.device) != self._request["runtime"]["device"] or str(p.dtype) != "torch.float32" for p in parameters)):
                raise ValueError("all actual trainable parameters require unique optimizer ownership")
            receipt = initial_reader(owner)
            if (not isinstance(receipt, dict) or receipt.get("schema") != "pg_canonical_" + kind + "_initial_owner_v1"
                    or receipt.get("task_id") != self.repeat["task_id"] or not _zero(receipt.get("completed_steps"))
                    or receipt.get("phase") != "ready" or receipt.get("restored") is not False
                    or type(receipt.get("seed")) is not int or receipt["seed"] != 0
                    or receipt.get("rng_version") != "forge-rng-v1"
                    or receipt.get("device") != self._request["runtime"]["device"]
                    or receipt.get("resolved_recipe_sha256") != self.repeat["effective_recipe_digest"]
                    or receipt.get("source_contract_sha256") != self.repeat["source_contract_sha256"]
                    or not receipt.get("object_bindings")
                    or any(value is not True for value in receipt["object_bindings"].values())):
                raise ValueError("actual initialization receipt disagrees with the source-bound repeat")
            for field in ("optimizer_updates", "optimizer_state_entries"):
                if not receipt.get(field) or any(not _zero(value) for value in receipt[field].values()):
                    raise ValueError("retained optimizer witness must contain only actual zero counts")
            if receipt["table"].get("shape") != list(owner.prior.z.shape) or receipt["table"].get("requires_grad") is not True:
                raise ValueError("actual task prior table is not witnessed")
            if "full_resolved_recipe" in receipt and digest(receipt["full_resolved_recipe"]) != self.repeat["effective_recipe_digest"]:
                raise ValueError("initial receipt changed the complete public Recipe")
            if kind in {"image", "mog"}:
                rng = receipt["global_rng"]
                _sha(rng["before_sha256"], "initial global RNG")
                if rng["before_sha256"] != rng["after_sha256"]:
                    raise ValueError("construction changed unowned global RNG")
                for value in receipt["model_sha256"].values():
                    _sha(value, "initialized actual model")
            else:
                _sha(receipt["initial_owner_state_sha256"], "actual initial AE state")
                _sha(receipt["initial_global_rng_sha256"], "actual initial AE global RNG")
            self._source_guard(); self._admission_guard()
            self._owner = owner
            self.initialization_witness = dict(schema="forge_canonical_full_atlas_initialization_witness_v1",
                repeat=deepcopy(self.repeat), initial_owner=deepcopy(receipt), factory_calls=1,
                source_checks=2, admission_checks=2, actual_policy_owners=list(required_owners),
                historical_checkpoint_loaded=False, accepted_numeric_credit=False)
            self._phase = "verified"
            return owner
        except BaseException:
            self._phase = "failed"
            raise

    def require_owned(self, owner):
        if self._phase != "verified" or owner is not self._owner:
            raise ValueError("the running owner has no admitted new-initialization witness")
        return deepcopy(self.initialization_witness)

# Source-derived metadata roster, not initialization or numerical credit.
ROSTER = {'img_stripes2': {'kind': 'image', 'task_digest': '1cc1016fabd7d6b1d3bd2948bf58c7df8c3c947fa58a18e2469f211c41b480b6', 'task_file_sha256': '010a96d3829b639d13f1150f30a5ff44119bcc10a23822f4191773bab177eb70', 'binding_sha256': '6209a9c25f10ccc65abaaacdc7ed42b33d599eba4489fedb9333512f62329e57', 'binding_schema': 'pg_canonical_image_policy_binding_v1', 'source_contract_sha256': 'd8c91bcfc7990bff18e5382cfb4d16ff282f436394ce54b80440d23da1f44de6', 'recipe_sha256': 'c64791736011b9eda199a66e89bf26e454184720e8df23393cc1d3dc39806eff', 'adapter_sha256': 'e8662a0ecf313fc768eea413496e8bc028666f762787cc6d0ae16535ed72588e', 'updates': 600, 'timeout_seconds': 1800, 'observations': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600], 'observation_contract_sha256': '79f762cdd8af020def7597ecb8c512a0336dafde395203210495ce8259762d05'}, 'img_bars4': {'kind': 'image', 'task_digest': '51e7d0412264e2b564b2dc9c884b8b094cbe86884dca2795032da25efe98d656', 'task_file_sha256': '562c05c07de33e5a5e088cff097af9cf9f6f8b37bb436aed37bf502fb7aff86f', 'binding_sha256': '55278a7663b5b30c6eb0db7c39d044f0c226003609f5b7658569ddcc8d450905', 'binding_schema': 'pg_canonical_image_policy_binding_v1', 'source_contract_sha256': '3156c9e6a5f855144bf613d98589133047c103f1266d3d1b8494adaa369d601f', 'recipe_sha256': 'c64791736011b9eda199a66e89bf26e454184720e8df23393cc1d3dc39806eff', 'adapter_sha256': 'e8662a0ecf313fc768eea413496e8bc028666f762787cc6d0ae16535ed72588e', 'updates': 600, 'timeout_seconds': 1800, 'observations': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600], 'observation_contract_sha256': '79f762cdd8af020def7597ecb8c512a0336dafde395203210495ce8259762d05'}, 'img_blobs4': {'kind': 'image', 'task_digest': '153361882e740907f408eb42796cf970160bd5d2a565aad62a468e2b52b913d1', 'task_file_sha256': '15c7ec3cfe80539b14d24824afc1a6fffae99e27b5a38e75c991823b5aceb336', 'binding_sha256': '31dad3e94ebe3c0ad305feae3c92b724d0c6e1cce2665746d6fe3219b321ce58', 'binding_schema': 'pg_canonical_image_policy_binding_v1', 'source_contract_sha256': '453d8c8c1d65b8709593603e151090c9c7bd12ff2079564cd8ce34ea39c9047c', 'recipe_sha256': 'c64791736011b9eda199a66e89bf26e454184720e8df23393cc1d3dc39806eff', 'adapter_sha256': 'e8662a0ecf313fc768eea413496e8bc028666f762787cc6d0ae16535ed72588e', 'updates': 600, 'timeout_seconds': 1800, 'observations': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600], 'observation_contract_sha256': '79f762cdd8af020def7597ecb8c512a0336dafde395203210495ce8259762d05'}, 'img_intensity2': {'kind': 'image', 'task_digest': '694752fc66c0a17d73afe9081fec8409f18c178ba252b2152431569427da6003', 'task_file_sha256': '6a69162bee7dd1c50c9f596f29ed339a5e475c296156c4324e5092099cddf90d', 'binding_sha256': 'f95aec7c96f210d944e49d06d460c0b753f5321030c8d5da3a4377379c394dbd', 'binding_schema': 'pg_canonical_image_policy_binding_v1', 'source_contract_sha256': 'e7545d25230835dcecaa95a96e3c5fe5947473404cb9b6b47d8cf91913f4bf6d', 'recipe_sha256': 'c64791736011b9eda199a66e89bf26e454184720e8df23393cc1d3dc39806eff', 'adapter_sha256': 'e8662a0ecf313fc768eea413496e8bc028666f762787cc6d0ae16535ed72588e', 'updates': 600, 'timeout_seconds': 1800, 'observations': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600], 'observation_contract_sha256': '79f762cdd8af020def7597ecb8c512a0336dafde395203210495ce8259762d05'}, 'ring16_acquisition': {'kind': 'mog', 'task_digest': '6858cca00f8efa1313da89a0e0dcb18d726ea593a5f44ebfb2f03a155523ec3c', 'task_file_sha256': 'e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1', 'binding_sha256': '96f67f8fb09b32555ef4d364f2ecb1083f1a809fbbc8a80b13c4bd29b3384390', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': 'bd0b8aae227745f864c0c588da16f9ac17ac1191eee8e1fc84c7cdd164d694ea', 'recipe_sha256': '10d2e8df9667cecde1a863dd098d50adaa77946a8e424366d96460822aaa83da', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 400, 'timeout_seconds': 300, 'observations': [17, 34, 50, 67, 84, 100, 117, 134, 150, 167, 184, 200, 217, 234, 250, 267, 284, 300, 317, 334, 350, 367, 384, 400], 'observation_contract_sha256': '113279d66bfc449cad293410c3203bd0727871794edccd2e56b30a265997ae68'}, 'mode_hold': {'kind': 'mog', 'task_digest': '284ee9b992caec120bac36eef2dc82b3b94409633e2ba9ed1f6cee1cec23d464', 'task_file_sha256': '79e11760a8851a65f3739775b5e4041bacb0a79050e8e21f52ad1eb9500435f5', 'binding_sha256': 'c3184e10b8c9fba7072d44cbbd15da6f194dc7b2b1d5c124f5880622110067d7', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': '2710d2b7575ce3c70a7b86a61becb9a918a9e6c0d0a99c4208273fe268e1cfe4', 'recipe_sha256': 'dba751ae2dd021484bb10a7b794cf8e1cad7fe8f27e3763e1fe47b607f75994e', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 1200, 'timeout_seconds': 1800, 'observations': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200], 'observation_contract_sha256': '55710deae0dabf99f14f1816a05825453903ffe8eb701cb50170eb6c4f393f2d'}, 'vector_two_broad': {'kind': 'mog', 'task_digest': 'dad52dad9d43f75a73ecbf07c78c75cd90e5c8de39102c459126cdcfe23cd20c', 'task_file_sha256': '7239390e9da217364f4142d0a90afd60e75317cd5fe362efb9b0e68d0e1bccab', 'binding_sha256': '6b0676d844a208836ac102e8cc6b91b5da915080bb0d93ceea78eba9d9283479', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': '992f6371a3fefe63bd97f7c38ac39a1eef41b8783ca2f1aa96d8c54dead5ff70', 'recipe_sha256': '10d2e8df9667cecde1a863dd098d50adaa77946a8e424366d96460822aaa83da', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 1200, 'timeout_seconds': 1800, 'observations': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200], 'observation_contract_sha256': '55710deae0dabf99f14f1816a05825453903ffe8eb701cb50170eb6c4f393f2d'}, 'vector_unequal_mass': {'kind': 'mog', 'task_digest': 'b0f07e1064fa9c13271f630bc93f4979625c5c96ba79930349c8c8a76b1ebe46', 'task_file_sha256': 'ea1b35df2bd0ab55b50d47ade00a758775d29130641fc1e76db28c03009164ff', 'binding_sha256': 'ff6593a46bf8d5468a7594958d9ca76ada6fe2d61e2ef87654528ef78a3a68d0', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': '2ba63b01c17f571b3c654548c461860c72fe21a0361f5d5e1710910f5e81cd41', 'recipe_sha256': '10d2e8df9667cecde1a863dd098d50adaa77946a8e424366d96460822aaa83da', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 1200, 'timeout_seconds': 1800, 'observations': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200], 'observation_contract_sha256': '55710deae0dabf99f14f1816a05825453903ffe8eb701cb50170eb6c4f393f2d'}, 'vector_unequal_width': {'kind': 'mog', 'task_digest': '7cc03a981745e52719fead313c50cd4e9f048dfa7e8cff55ca30a9d22889ca15', 'task_file_sha256': 'e3f18fe65e198563908aa1b2b68a76bf46398427037cddffef4b42e0cd3d0c8f', 'binding_sha256': '63e03a734989a666b43aa6c1c0163ea1d4d28c85c830777a8f1272743abf23dc', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': '5741f2041b2055e6ef08c85e4af4337177530544ce1adc8f7eff98a9003c38bb', 'recipe_sha256': '10d2e8df9667cecde1a863dd098d50adaa77946a8e424366d96460822aaa83da', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 1200, 'timeout_seconds': 1800, 'observations': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200], 'observation_contract_sha256': '55710deae0dabf99f14f1816a05825453903ffe8eb701cb50170eb6c4f393f2d'}, 'vector_anisotropic': {'kind': 'mog', 'task_digest': 'ce96f89823835509b1869ae6aff565270b06cf5e2cd018c6cf946e587352d0f5', 'task_file_sha256': 'e98512a93ba2d3a10a5fe6640cf74c5d22f88f917e9a5cf313302cc4aed1a24b', 'binding_sha256': '20a43779e959434bfcff6215db8bd08c1c70e6abb0034dbcab2624de414482c9', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': '8ff0a57e3823b161cd704174dce0d5ae094fd3bd9375aec6fa191727a0371ae0', 'recipe_sha256': '10d2e8df9667cecde1a863dd098d50adaa77946a8e424366d96460822aaa83da', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 1200, 'timeout_seconds': 1800, 'observations': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200], 'observation_contract_sha256': '55710deae0dabf99f14f1816a05825453903ffe8eb701cb50170eb6c4f393f2d'}, 'vector_overlap': {'kind': 'mog', 'task_digest': '72c2d16e1e1081365728af778955991bb622811ede0ed782466d03ca3eefe717', 'task_file_sha256': '72bde8ab5f9a5222b095ee17849c27823156bd28675f443138d56542c9e69c35', 'binding_sha256': 'f747cea6d145f65553e4d787b1aadcbe6193cd37e2b83475fabf1824470aa203', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': 'd9a3b1eaad9a11a430de10bb85100d6858803411074e55aaf3c6516cf792ed73', 'recipe_sha256': '10d2e8df9667cecde1a863dd098d50adaa77946a8e424366d96460822aaa83da', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 1200, 'timeout_seconds': 1800, 'observations': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200], 'observation_contract_sha256': '55710deae0dabf99f14f1816a05825453903ffe8eb701cb50170eb6c4f393f2d'}, 'vector_spiral': {'kind': 'mog', 'task_digest': '8bf5d41cd5f8b149161178370f3df32249e7d39dc6aa0839100da37f34e6b628', 'task_file_sha256': 'd47cf689d77375873c25b5d6aa3610e9d6250ee5a9d2f2592538d49da943ff9f', 'binding_sha256': 'e47a44f085859174bd20d7535f24d520d2f14c7f319038604d7dc282d96d2bc5', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': 'bda36341dece41cf2e3f39fed92d171031efa43a3cf80260c33e55f96ec3ed0d', 'recipe_sha256': '10d2e8df9667cecde1a863dd098d50adaa77946a8e424366d96460822aaa83da', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 1600, 'timeout_seconds': 1800, 'observations': [67, 134, 200, 267, 334, 400, 467, 534, 600, 667, 734, 800, 867, 934, 1000, 1067, 1134, 1200, 1267, 1334, 1400, 1467, 1534, 1600], 'observation_contract_sha256': '3825c0f865fa4aa66edaec53e96ce5a0980e5f12b7d0a7826b2e1e4cbbb71553'}, 'grid100': {'kind': 'mog', 'task_digest': 'e1b414bd92140a95338f0bc62d6c5328efa919e2913457a258ac2de33daeac3f', 'task_file_sha256': '64709b1544155fd3e19ac1aef96453e2183145cc2be029e7da5df3bfb4b291b5', 'binding_sha256': 'a39febfabae601388b359d145100a6fca3b4bc0a60bcdad2a74958430438e485', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': '9734f5a9699c9bc40dbbe7aa60d7a12cd7a674339905f1af95b146c639b228fe', 'recipe_sha256': 'a186235341b37123e9944253a40ed2105993354954ca37f6b527f2cae44a1132', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 7000, 'timeout_seconds': 3600, 'observations': [0, 1, 10, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000], 'observation_contract_sha256': '5d81241b12ef770224892f4686312cc6722dba0dbf81242693e90f75344e1a2b'}, 'rotated100': {'kind': 'mog', 'task_digest': '282ecf1465be0d861b1d500249fb1aadc0303310f74e9b8ec9b7240684e274ed', 'task_file_sha256': 'f8691aa4152885a34dc4e32eda6bbbc5d7df2699cc7b582f9bc25c091ea9c35d', 'binding_sha256': '6a9069332d927a934b342cf020d4676eb100acacf77430a2f4ec30ca85408ce5', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': 'c5c7d65c77185f3902cd2b90e76596b360a913d92d7b43ec6829eefc81f88d05', 'recipe_sha256': 'a186235341b37123e9944253a40ed2105993354954ca37f6b527f2cae44a1132', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 7000, 'timeout_seconds': 3600, 'observations': [0, 1, 10, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000], 'observation_contract_sha256': '5d81241b12ef770224892f4686312cc6722dba0dbf81242693e90f75344e1a2b'}, 'staggered100': {'kind': 'mog', 'task_digest': '278129827827015bd53d13f1b7136f37eaafc9cbf34f46fcbf531bfb7eb5ba72', 'task_file_sha256': '5c39cf3faf007d146f861b9962965b92ec17bcf7d9bd906aa9f1f0d21c82d4a1', 'binding_sha256': 'f0fbbe65e16d63fdbb1f26bce43c30e29e24a41c8e122161c03aa33e470b8ca1', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': 'ac6ae7606ead7648c6dfc813cb7b5a87041529d7535b4c9cb62f8f3802a63561', 'recipe_sha256': 'a186235341b37123e9944253a40ed2105993354954ca37f6b527f2cae44a1132', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 7000, 'timeout_seconds': 3600, 'observations': [0, 1, 10, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000], 'observation_contract_sha256': '5d81241b12ef770224892f4686312cc6722dba0dbf81242693e90f75344e1a2b'}, 'ring_hold': {'kind': 'mog', 'task_digest': 'f38def7f12378726a38de16eb8420efe04a086ae395c48dfa1c29e9c9702c264', 'task_file_sha256': '7d3e33f19db943f6f55016b9cbf2fe93221cc3e202c4c86190d5ff981b1fdfc9', 'binding_sha256': '9c83822d438eb37f28017aa009a6f4966f490f887bd5391b2df29b08198cafe3', 'binding_schema': 'pg_canonical_mog_policy_binding_v1', 'source_contract_sha256': '09218db797e4e74896e66fc96d5880c1520a94ca51cc116753169c1140388424', 'recipe_sha256': 'dba751ae2dd021484bb10a7b794cf8e1cad7fe8f27e3763e1fe47b607f75994e', 'adapter_sha256': '751f072f7cc6249e79e83c7df6e0b0a09170bb9f0ae9ca0a2df12e72f97c08e9', 'updates': 7500, 'timeout_seconds': 3600, 'observations': None, 'observation_contract_sha256': '27e355485aea23435546902dbd0c09f0f6d336c51738ca2102451f14e583d35d'}, 'ae_gan_hold': {'kind': 'ae', 'task_digest': 'd7ce55cf6c7679fdb2dc7a5292bc7dfdb964f602daec09a39610599d20e4e103', 'task_file_sha256': '53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79', 'binding_sha256': '7b5fe6caba38f42b0408b244246381866b5e0d080a8dd9ee694293fc2ad299d6', 'binding_schema': 'pg_canonical_ae_policy_binding_v1', 'source_contract_sha256': '0e88d7f326debe16b98148053ec734bf802627cca5484fe397860464d3ebad14', 'recipe_sha256': 'a26eb000fbb7616c90265a4bf31c5126e75f86a4cbd04912d71dc0292167e250', 'adapter_sha256': 'e09306a6ab520939d38651d8ea707bb900b237381f2144d15498d9b5ad6364e5', 'updates': 250, 'timeout_seconds': 300, 'observations': [11, 21, 32, 42, 53, 63, 73, 84, 94, 105, 115, 125, 136, 146, 157, 167, 178, 188, 198, 209, 219, 230, 240, 250], 'observation_contract_sha256': 'e65063c0176344c0d423276d5e56b729601cd7b6b1909e56cd6e018a34b813b9'}}
