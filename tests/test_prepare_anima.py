import pytest

from experiments.prepare_anima_transplant import select_tensors


@pytest.mark.parametrize('root', ['net.', 'model.diffusion_model.'])
def test_export_namespaces_select_same_slice(root):
    wanted = ['blocks.0.self_attn.q_proj.weight', 'blocks.1.mlp.layer1.weight',
              't_embedder.1.linear_1.weight', 't_embedding_norm.weight']
    header = {root + k: {'shape': [2]} for k in wanted}
    header[root + 'blocks.10.mlp.layer1.weight'] = {'shape': [3]}
    header[root + 'llm_adapter.weight'] = {'shape': [4]}
    selected_root, selected = select_tensors(header, [0, 1])
    assert selected_root == root
    assert {k.removeprefix(root) for k in selected} == set(wanted)
    assert all(v == {'shape': [2]} for v in selected.values())


def test_missing_or_ambiguous_exports_rejected():
    groups = ['blocks.0.weight', 't_embedder.weight', 't_embedding_norm.weight']
    with pytest.raises(ValueError, match='Missing or ambiguous'):
        select_tensors({'net.blocks.0.weight': {}}, [0])
    with pytest.raises(ValueError, match='Missing or ambiguous'):
        select_tensors({root + k: {} for root in ('net.', 'model.diffusion_model.')
                        for k in groups}, [0])
