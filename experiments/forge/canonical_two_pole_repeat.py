"""One recognized fresh two-pole intent and its actual construction witness.

Declarations distinguish admission keys. They never assert that initialization
occurred; only construct() witnesses the source-bound owner factory in the
admitted child. This module imports no scientific package and reads no files.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json


PACKET_SCHEMA = "pg_canonical_two_pole_first_case_v1"
CASE_ID = "canonical-two-pole-full-atlas-ember552-v1"
REPEAT_ID = "canonical-common26-two-pole-ember552-fresh-init-seed0-v1"
REPEAT_SCHEMA = "forge_canonical_two_pole_fresh_repeat_v1"
TASK_FILE_SHA256 = "55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5"
TASK_DIGEST = "2f0207310d6bb7b290bdc520d7992eb4e6da411becae69a76d76b1232897db8b"
CONFIG_FILE_SHA256 = "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"
CONFIG_DIGEST = "6ce21adbd06cb20a1e9725876cf19025f7bb507a7c7ffeba6bf95b2c1f5fdfec"
PROTOCOL_DIGEST = "7b975bab8008c880a88e04d9e9bbdf4d1ab549d46d1d894b9e42369fdc0931b2"
PROTOCOL_FILE_SHA256 = "3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803"
RECIPE_FILE_SHA256 = "1a7f9df746e242a2774068c819d70d50cd72dc4f7f6b22b0bd13e1345dcf8ab2"
EFFECTIVE_RECIPE_DIGEST = "e6774c620e267474e6db4cf227b80fa8ee72b7f3c8ef2f352cd2789335f70b3f"
MODULE = "experiments/forge/canonical_two_pole_repeat.py"
ADAPTER = "experiments/forge/canonical_two_pole_adapter.py"
_CONSTRUCTIONS = []  # Private references prevent identity reuse within a child.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid " + label)
    return value


def _task(value):
    task = deepcopy(value)
    if not isinstance(task, dict):
        raise ValueError("fresh repeat requires its canonical task")
    if "preflight_blockers" in task:
        blockers = task.pop("preflight_blockers")
        if not isinstance(blockers, list) or blockers:
            raise ValueError("a blocked or malformed compiled task cannot become a fresh attempt")
    if "field_ownership" in task and not isinstance(task.pop("field_ownership"), dict):
        raise ValueError("malformed compiled ownership annotation")
    if digest(task) != TASK_DIGEST:
        raise ValueError("fresh repeat changed the canonical two-pole question")
    return task


def base_protocol(value):
    protocol = deepcopy(value)
    if not isinstance(protocol, dict):
        raise ValueError("fresh repeat requires the common screening protocol")
    protocol.pop("scientific_repeat", None)
    if digest(protocol) != PROTOCOL_DIGEST:
        raise ValueError("fresh repeat changed the ordinary screening protocol")
    return protocol


def make_scientific_repeat(request):
    """Bind a frozen intent to actual request metadata, without freshness credit."""
    task = _task(request["task"])
    protocol = base_protocol(request["protocol"])
    candidate, binding, source = request["candidate"], request["binding"], request["source"]
    if (candidate.get("trainer_family") != "atlas"
            or digest(candidate.get("recipe_overrides")) != CONFIG_DIGEST):
        raise ValueError("fresh repeat requires the complete original Atlas configuration")
    if (binding.get("schema") != "pg_canonical_two_pole_policy_binding_v1"
            or binding.get("task_sha256") != TASK_FILE_SHA256
            or binding.get("config_sha256") != CONFIG_FILE_SHA256):
        raise ValueError("fresh repeat lacks the source-derived canonical binding")
    contract_hash = _sha(binding.get("source_contract_sha256"), "source contract digest")
    if contract_hash != digest(binding["source_contract"]):
        raise ValueError("fresh repeat source-contract bytes disagree")
    recipe = binding["recipe"]
    if not isinstance(recipe, dict) or digest(recipe) != EFFECTIVE_RECIPE_DIGEST:
        raise ValueError("fresh repeat changed the legal full-Atlas direct-coordinate binding")
    origin = source.get("origin_commit")
    if (not isinstance(origin, str) or len(origin) != 40
            or any(c not in "0123456789abcdef" for c in origin)):
        raise ValueError("fresh repeat requires an exact source origin")
    files = source.get("files", {})
    for member in (MODULE, ADAPTER, "experiments/forge/policy_execution.py",
                   "configs/forge/tasks/two_pole.json", "configs/100gaussians/atlas.json",
                   "configs/forge/protocols/screening.json", "benchmarks/locked_shared/two_pole.py",
                   "benchmarks/locked_shared/observation.py", "benchmarks/transfer_suite/protocol.py",
                   "particlegan/recipes.py", "particlegan/policy.py", "experiments/forge/rng.py"):
        _sha(files.get(member), "source member " + member)
    fixed = {"configs/forge/tasks/two_pole.json": TASK_FILE_SHA256,
             "configs/100gaussians/atlas.json": CONFIG_FILE_SHA256,
             "configs/forge/protocols/screening.json": PROTOCOL_FILE_SHA256,
             "particlegan/recipes.py": RECIPE_FILE_SHA256}
    if any(files[member] != expected for member, expected in fixed.items()):
        raise ValueError("fresh repeat changed canonical declaration or Recipe source bytes")
    if files[ADAPTER] != binding.get("adapter_sha256"):
        raise ValueError("fresh repeat adapter source is not its declared source member")
    return {"schema": REPEAT_SCHEMA, "repeat_id": REPEAT_ID, "case_id": CASE_ID,
            "task_id": task["id"], "authority_sequence": 552,
            "seed": protocol["seed"], "rng_version": protocol["rng"]["derivation"],
            "task_digest": digest(task), "protocol_digest": digest(protocol),
            "candidate_digest": digest(candidate), "config_sha256": CONFIG_FILE_SHA256,
            "effective_recipe_digest": digest(recipe), "source_contract_sha256": contract_hash,
            "source_origin_commit": origin, "source_digest": _sha(source.get("digest"), "source digest"),
            "runtime_digest": digest(request["runtime"]), "initialization": "new_owner_factory",
            "execution_updates": 80, "recipe_schedule_horizon": None}


def validate_scientific_repeat(request):
    repeat = request["protocol"].get("scientific_repeat")
    if not isinstance(repeat, dict) or digest(repeat) != digest(make_scientific_repeat(request)):
        raise ValueError("missing or altered recognized first-case scientific repeat")
    return deepcopy(repeat)


def coordinator_repeat(packet, trial, row):
    """Return an attempt-key binding only for the recognized one-case envelope."""
    schema = packet.get("schema", "")
    claim = (isinstance(schema, str) and schema.startswith("pg_canonical_two_pole_first_case"))
    claim = claim or str(row.get("id", "")).startswith("canonical-two-pole-full-atlas-")
    claim = claim or "scientific_repeat" in packet or "scientific_repeat" in packet.get("protocol", {})
    if not claim:
        return None
    if schema != PACKET_SCHEMA or row.get("id") != CASE_ID:
        raise ValueError("unknown or foreign first-case repeat envelope")
    request = packet["request"]
    repeat = validate_scientific_repeat(request)
    expected_row = {"id": CASE_ID, "task_id": "two_pole", "timeout_seconds": 300,
                    "allowance_seconds": 300, "status": "NOT_RUN"}
    expected_case = {"task": request["task"], "candidate": request["candidate"],
                     "protocol": request["protocol"], "binding": request["binding"]}
    if (digest(row) != digest(expected_row) or digest(packet.get("rows")) != digest([expected_row])
            or digest(packet.get("case_definitions")) != digest({CASE_ID: expected_case})
            or digest(packet.get("lane_runtime")) != digest(request["runtime"])
            or type(packet["spec"].get("frames")) is not int or packet["spec"]["frames"] != 24):
        raise ValueError("first-case container cannot create another identity or carry historical credit")
    if (packet.get("scientific_repeat") != repeat or packet.get("protocol") != request["protocol"]
            or packet["spec"].get("scientific_repeat") != repeat
            or packet["execution_source"]["digest"] != repeat["source_digest"]
            or packet["execution_source"]["origin_commit"] != repeat["source_origin_commit"]
            or packet["execution_source"]["files"] != request["source"]["files"]
            or trial.get("family") != "atlas"
            or digest(trial.get("recipe_overrides")) != digest(request["candidate"]["recipe_overrides"])
            or type(row.get("timeout_seconds")) is not int or row["timeout_seconds"] != 300
            or packet["spec"].get("export_grace_seconds") != 0
            or packet["spec"].get("retries") != 0):
        raise ValueError("first-case repeat disagrees with admission metadata")
    return repeat


def _public_type(value, module, name):
    if type(value).__module__ != module or type(value).__name__ != name:
        raise ValueError("fresh construction requires actual " + module + "." + name)


class FreshRepeatGuard:
    """Witness one source-bound actual owner construction inside real admission.

    The callbacks are supplied by the source-bound child controller, not by a
    JSON receipt. Neither a cached result nor a caller's fresh flag can enter
    this interface. Model-free controls supply explicit private test doubles.
    """
    def __init__(self, request, *, source_guard, admission_guard):
        if not callable(source_guard) or not callable(admission_guard):
            raise TypeError("fresh construction requires source and admission guards")
        self.repeat = validate_scientific_repeat(request)
        self._request = deepcopy(request)
        self._source_guard, self._admission_guard = source_guard, admission_guard
        self._phase, self._owner, self.initialization_witness = "pending", None, None

    def construct(self, factory, initial_reader):
        if self._phase != "pending":
            raise RuntimeError("one fresh repeat permits one construction and no retry")
        if not callable(factory) or not callable(initial_reader):
            raise TypeError("fresh construction requires its actual factory and owner reader")
        self._phase = "constructing"
        try:
            self._source_guard()
            self._admission_guard()
            owner = factory()
            _public_type(owner, "experiments.forge.canonical_two_pole_adapter", "TwoPoleOwner")
            _public_type(owner.policy, "particlegan.policy", "UpdatePolicy")
            _public_type(owner.critic, "benchmarks.locked_shared.two_pole", "HostCritic")
            _public_type(owner.generator, "torch.nn.modules.linear", "Identity")
            _public_type(owner.opt_g, "particlegan.k3p", "K3PGeneratorAdam")
            _public_type(owner.opt_d, "particlegan.ka2", "KA2CriticAdam")
            _public_type(owner.recipe, "particlegan.recipes", "Recipe")
            _public_type(owner.table, "torch.nn.parameter", "Parameter")
            _public_type(owner.streams, "experiments.forge.rng", "NamedStreams")
            token = owner.construction_id
            if (type(token) is not object or owner.restored is not False
                    or any(owner is old or token is old_token for old, old_token in _CONSTRUCTIONS)):
                raise ValueError("cached or restored owner is not a new initialization")
            _CONSTRUCTIONS.append((owner, token))
            if (type(owner.policy.completed_steps) is not int or owner.policy.completed_steps != 0
                    or owner.policy._phase != "ready" or owner.opt_g.state or owner.opt_d.state
                    or owner.policy.table is not owner.table or owner.policy.D is not owner.critic
                    or owner.policy.G is not owner.generator or owner.policy.opt_g is not owner.opt_g
                    or owner.policy.opt_d is not owner.opt_d or owner.policy.recipe is not owner.recipe
                    or owner.policy.table_optimizer is not owner.opt_g
                    or digest(asdict(owner.recipe)) != self.repeat["effective_recipe_digest"]
                    or tuple(owner.table.shape) != (12, 1) or owner.table.requires_grad is not True
                    or str(owner.table.device) != "cpu"
                    or any(value != 0.0 for row in owner.table.detach().tolist() for value in row)
                    or owner.streams.seed != 0 or owner.streams.version != "forge-rng-v1"):
                raise ValueError("actual owner is not the canonical zero-update starting state")
            if initial_reader.__module__ != "experiments.forge.canonical_two_pole_adapter" \
                    or initial_reader.__name__ != "owner_initial_receipt":
                raise ValueError("fresh construction requires the source-bound owner reader")
            receipt = initial_reader(owner)
            required = {"schema", "completed_steps", "phase", "restored", "seed", "rng_version",
                        "table", "critic", "generator", "optimizer_updates", "optimizer_state_entries",
                        "object_bindings", "resolved_recipe_sha256", "source_contract_sha256"}
            if not isinstance(receipt, dict) or set(receipt) != required:
                raise ValueError("invalid actual initial-owner receipt")
            aliases = {"policy_table", "policy_critic", "policy_generator", "table_optimizer",
                       "unique_parameter_ownership"}
            if (receipt["schema"] != "pg_canonical_two_pole_initial_owner_v1"
                    or type(receipt["completed_steps"]) is not int or receipt["completed_steps"] != 0
                    or receipt["phase"] != "ready" or receipt["restored"] is not False
                    or type(receipt["seed"]) is not int or receipt["seed"] != 0
                    or receipt["rng_version"] != "forge-rng-v1"
                    or receipt["table"] != {"shape": [12, 1], "all_zero": True, "requires_grad": True}
                    or receipt["generator"] != {"type": "torch.nn.Identity", "trainable_parameters": 0}
                    or set(receipt["object_bindings"]) != aliases
                    or any(value is not True for value in receipt["object_bindings"].values())
                    or receipt["resolved_recipe_sha256"] != self.repeat["effective_recipe_digest"]
                    or receipt["source_contract_sha256"] != self.repeat["source_contract_sha256"]):
                raise ValueError("initial-owner witness disagrees with the frozen repeat")
            critic = receipt["critic"]
            if (set(critic) != {"type", "stored_weights_exact", "state_sha256"}
                    or critic["type"] != "benchmarks.locked_shared.two_pole.HostCritic"
                    or critic["stored_weights_exact"] is not True):
                raise ValueError("new critic did not retain the stored source initialization")
            _sha(critic["state_sha256"], "initial critic state")
            for key, roles in (("optimizer_updates", {"prior", "discriminator", "noise"}),
                               ("optimizer_state_entries", {"prior", "discriminator"})):
                if (set(receipt[key]) != roles
                        or any(type(value) is not int or value != 0 for value in receipt[key].values())):
                    raise ValueError("initial optimizer state is not empty")
            self._source_guard()
            self._admission_guard()
            self._owner = owner
            self.initialization_witness = {"schema": "forge_canonical_two_pole_initialization_witness_v1",
                "repeat": deepcopy(self.repeat), "initial_owner": deepcopy(receipt),
                "factory_calls": 1, "source_checks": 2, "admission_checks": 2,
                "historical_checkpoint_loaded": False, "accepted_numeric_credit": False}
            self._phase = "verified"
            return owner
        except BaseException:
            self._phase = "failed"
            raise

    def require_owned(self, owner):
        if self._phase != "verified" or owner is not self._owner:
            raise ValueError("the running owner has no actual fresh-construction witness")
        return deepcopy(self.initialization_witness)
