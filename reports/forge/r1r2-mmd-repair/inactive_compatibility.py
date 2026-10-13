"""Run the frozen inactive comparator with two checked added default fields.

The original comparator and conftest bytes remain frozen. Only their existing
scoped metadata adapter's exact-default map is extended in memory; the adapter
retains all original tensor, optimizer, update and consumed-stream assertions.
This is a bounded software cohort, not scientific qualification.
"""
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-mmd-preparation-20261012')
OLD = '        "energy_distance_weight": 0.0}'
NEW = ('        "energy_distance_weight": 0.0,\n'
       '        "output_mmd_weight": 0.0,\n'
       '        "output_mmd_bandwidths": (.0625, .125, .25, .5, 1., 2., 4.)}')


def _check_inactive_ownership(name, default, ownership, prior_binding):
    """Preserve the frozen ownership check; accept exact JSON widths only."""
    if ownership.get("status") == "not_applicable":
        assert name == "prior_proximal_strength" and default == 0.0
        assert prior_binding["latent_table_controls"] is False
        assert ownership == dict(value=None, owner="hyperparameter",
            source="task prior-control applicability", status="not_applicable",
            reference_recipe_value=default, representation=prior_binding["representation"])
    else:
        value = ownership["value"]
        if name == "output_mmd_bandwidths" and isinstance(value, list):
            assert all(type(item) is float for item in value)
            value = tuple(value)
        assert value == default
        assert ownership["source"].startswith("public Recipe ")


class ScopedDefaults:
    def pytest_configure(self, config):
        matches = [module for module in config.pluginmanager.get_plugins()
                   if getattr(module, '__file__', None)
                   and Path(module.__file__).resolve() == ROOT / 'tests/conftest.py']
        assert len(matches) == 1
        self.module = matches[0]
        self.original = self.module._archived_inactive_comparison_scope
        source = inspect.getsource(self.original)
        assert source.count(OLD) == 1
        adapted = source.replace(OLD, NEW)
        assert adapted.replace(NEW, OLD) == source
        hook = '    adapted, diff = helper.adapter_source(original)\n'
        assert adapted.count(hook) == 1
        # The frozen helper emits JSON scalar literals. A tuple-valued added
        # default needs a Python literal; retain its complete AST proof that
        # only the metadata map changed, including all numerical assertions.
        literal_hook = (
            '    serializer = inspect.getsource(helper.adapter_source)\n'
            '    assert serializer.count("{json.dumps(value)}") == 1\n'
            '    serializer = serializer.replace("{json.dumps(value)}", "{repr(value)}")\n'
            '    helper_namespace = dict(helper.__dict__)\n'
            '    exec(compile(serializer, "inactive-mmd-literal-adapter", "exec"), helper_namespace)\n'
            '    adapted, diff = helper_namespace["adapter_source"](original)\n')
        adapted = adapted.replace(hook, literal_hook)
        namespace = dict(inspect.unwrap(self.original).__globals__)
        namespace['_check_inactive_ownership'] = _check_inactive_ownership
        exec(compile(adapted, str(__file__), 'exec'), namespace)
        self.module._archived_inactive_comparison_scope = namespace['_archived_inactive_comparison_scope']

    def pytest_unconfigure(self, config):
        if hasattr(self, 'original'):
            self.module._archived_inactive_comparison_scope = self.original


if __name__ == '__main__':
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    raise SystemExit(pytest.main([
        '-q', str(ROOT / 'tests/test_bcap_integration_compatibility.py'),
        f'--junitxml={ARCHIVE / "inactive-compatibility.xml"}',
    ], plugins=[ScopedDefaults()]))
