"""Narrow, checked allocation/cell metadata edits on an unchanged count law."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
DEFAULT_BASE=HERE.parents[3]/'integration/review/training-regression/joint-count/pkg-joint-count'
METHODS=('_ordinary_mass_transport','_ordinary_support_transport','_ordinary_global_transport')


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def ast_digest(node):return digest(ast.dump(node,include_attributes=False))


def node_source(text,node):
    lines=text.splitlines(keepends=True)
    start=min([node.lineno]+[n.lineno for n in node.decorator_list])
    return ''.join(lines[start-1:node.end_lineno])


def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError('Expected exactly one unchanged block: '+repr(before[:90]))
    return text.replace(before,after,1)


def batch_method(text):
    text=replace_once(text,
'''        within_deaths,within_births = torch.zeros_like(death_capacity),torch.zeros_like(birth_capacity)
        for group in range(self.mass_groups):
            members = groups==group
            within_deaths += _integer_allocate(death_capacity*members,int(within[group]))
            within_births += _integer_allocate(birth_capacity*members,int(within[group]))
''',
'''        within_deaths = _group_integer_allocate(death_capacity,groups,within)
        within_births = _group_integer_allocate(birth_capacity,groups,within)
''')
    text=replace_once(text,
'''        cross_deaths,cross_births = torch.zeros_like(death_capacity),torch.zeros_like(birth_capacity)
        for group in range(self.mass_groups):
            members = groups==group
            cross_deaths += _integer_allocate(remaining_death*members,int(cross_group_death[group]))
            cross_births += _integer_allocate(remaining_birth*members,int(cross_group_birth[group]))
''',
'''        cross_deaths = _group_integer_allocate(remaining_death,groups,cross_group_death)
        cross_births = _group_integer_allocate(remaining_birth,groups,cross_group_birth)
''')
    text=replace_once(text,
'''        deaths,births = within_deaths+cross_deaths,within_births+cross_births
''',
'''        deaths,births = within_deaths+cross_deaths,within_births+cross_births
        # One bounded metadata transfer preserves the original cell/RNG order.
        death_rows,birth_rows,within_death_rows,within_birth_rows,pool_rows,group_rows = torch.stack(
            (deaths,births,within_deaths,within_births,pool_counts,groups)).cpu().tolist()
        group_cells = [[] for _ in range(self.mass_groups)]
        for cell,group in enumerate(group_rows):
            group_cells[group].append(cell)
''')
    text=replace_once(text,'count = int(deaths[cell])','count = death_rows[cell]')
    text=replace_once(text,'count = int(births[cell])','count = birth_rows[cell]')
    text=replace_once(text,'torch.randperm(int(pool_counts[cell]),','torch.randperm(pool_rows[cell],')
    text=replace_once(text,
'''            for cell in (groups==group).nonzero().flatten():
                index = int(cell)
                children.append(children_by_cell[index][:int(within_deaths[index])])
                parents.append(parents_by_cell[index][:int(within_births[index])])
''',
'''            for index in group_cells[group]:
                children.append(children_by_cell[index][:within_death_rows[index]])
                parents.append(parents_by_cell[index][:within_birth_rows[index]])
''')
    text=replace_once(text,'children_by_cell[cell][int(within_deaths[cell]):]',
                           'children_by_cell[cell][within_death_rows[cell]:]')
    text=replace_once(text,'parents_by_cell[cell][int(within_births[cell]):]',
                           'parents_by_cell[cell][within_birth_rows[cell]:]')
    return text


def build(base):
    tree=ast.parse(base)
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellSnapshot')
    proposal=base;splices=[]
    for node in owner.body:
        if not isinstance(node,ast.FunctionDef) or node.name not in METHODS:continue
        before=node_source(base,node);after=batch_method(before)
        proposal=replace_once(proposal,before,after)
        changed=ast.parse('class FeatureCellSnapshot:\n'+after).body[0].body[0]
        splices.append(dict(kind='method',owner='FeatureCellSnapshot',name=node.name,
                           base_ast_sha256=ast_digest(node),proposal_ast_sha256=ast_digest(changed)))
    if len(splices)<2:raise ValueError('Expected unchanged mass/support count methods')
    prototype=(HERE/'pkg-PLAN/particlegan/feature_cells.py').read_text()
    helper=next(n for n in ast.parse(prototype).body if isinstance(n,ast.FunctionDef) and n.name=='_group_integer_allocate')
    helper_source=node_source(prototype,helper)+'\n\n'
    if any(isinstance(n,ast.FunctionDef) and n.name==helper.name for n in tree.body):
        raise ValueError('Helper already exists in reference source')
    proposal=replace_once(proposal,'class FeatureCellSnapshot:\n',helper_source+'class FeatureCellSnapshot:\n')
    splices.insert(0,dict(kind='function',name=helper.name,base_ast_sha256=None,
                         proposal_ast_sha256=ast_digest(helper)))
    ast.parse(proposal)
    return proposal,splices,digest(helper_source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package-root',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--proposal-package-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.proposal_package_root.exists() or args.output.exists():raise RuntimeError('Private output already exists')
    source=args.reference_package_root/'particlegan/feature_cells.py'
    original=source.read_text();proposal,splices,helper_sha=build(original)
    shutil.copytree(args.reference_package_root,args.proposal_package_root,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (args.proposal_package_root/'particlegan/feature_cells.py').write_text(proposal)
    args.output.write_text(json.dumps(dict(reference_package_root=str(args.reference_package_root.resolve()),
        proposal_package_root=str(args.proposal_package_root.resolve()),base_source_sha256=digest(original),
        proposal_source_sha256=digest(proposal),helper_source_sha256=helper_sha,ast_splices=splices,
        script_sha256=digest(Path(__file__).read_text())),indent=2)+'\n')


if __name__=='__main__':main()
