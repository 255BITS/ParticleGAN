"""Snapshot only modified legacy functions; retain their full original bodies."""
import ast,json
from pathlib import Path
root=Path(__file__).resolve().parent
modules=['mode_hold','trajectory','two_pole','hosts.unipolar','hosts.ae_gan_hold','hosts.cover_leftover','hosts.unused_token_hold','hosts.mid_scale_identity','hosts.residual_student']
receipts=[]
def players(node):
    return {n.func.value.id for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' and isinstance(n.func.value,ast.Name)}
class Expand(ast.NodeTransformer):
    count=0
    def visit_For(self,node):
        # Training loops are identified by optimizer calls, never task/evaluation values.
        roles=players(node)
        if 'opt_d' not in roles or not roles.intersection({'opt_g','opt_p'}):
            return self.generic_visit(node)
        if any(isinstance(n,ast.For) and n is not node and 'opt_d' in players(n) for n in ast.walk(node)):
            return self.generic_visit(node)
        end=max(i for i,s in enumerate(node.body) if players(s).intersection({'opt_g','opt_p'}))+1
        start=0
        # Noise amplitudes are set once at the original absolute outer clock.
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='set_step' for n in ast.walk(node.body[0])):
            start=1
        prefix=node.body[start:end]
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checkpoint' for stmt in prefix for n in ast.walk(stmt))
        inner=ast.For(target=ast.Name(id='_game_pass',ctx=ast.Store()),iter=ast.Call(func=ast.Attribute(value=ast.Name(id='_game_correction',ctx=ast.Load()),attr='passes',ctx=ast.Load()),args=[],keywords=[]),body=prefix,orelse=[])
        node.body=node.body[:start]+[inner]+node.body[end:]
        self.count+=1
        return node
for suffix in modules:
    module='benchmarks.locked_shared.'+suffix
    relative=Path(*module.split('.')).with_suffix('.py')
    source=root/'prepared/repos/cuda'/relative
    tree=ast.parse(source.read_text());selected=[];total=0
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            transform=Expand();changed=transform.visit(node)
            if transform.count:
                selected.append(changed);total+=transform.count
    assert total==1,(module,total)
    output=root/'candidates/extragradient/sources'/relative
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(ast.unparse(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])))+'\n')
    receipts.append(dict(module=module,source=str(relative),functions=[n.name for n in selected],expanded_loops=total))
(root/'candidates/extragradient/transforms.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps({'legacy_modules':len(receipts),'expanded_loops':sum(r['expanded_loops'] for r in receipts)}))
