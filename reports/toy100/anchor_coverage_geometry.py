"""Exact rational checks for a distinct-anchor support objective.

This is a finite one-dimensional geometry check, not a neural trainer or a
data clustering algorithm. It checks all quadratic branches for N=3,K=2
and retains an N=K counterexample to naive branch-stationarity claims.
"""

import argparse
from fractions import Fraction as F
from itertools import permutations, product
import hashlib
import json
from pathlib import Path


def assignments(n, k):
    return tuple(permutations(range(n), k))


def branch_loss(points, centers, assignment, nearest):
    n, k = len(points), len(centers)
    return (sum((points[j]-centers[a])**2 for a,j in enumerate(assignment))/k
            + sum((y-centers[b])**2 for y,b in zip(points,nearest))/n)


def anchor_cost(points, centers, assignment):
    return sum((points[j]-centers[a])**2 for a,j in enumerate(assignment))/len(centers)


def objective(points, centers):
    return (min(anchor_cost(points,centers,a) for a in assignments(len(points),len(centers)))
            + sum(min((y-c)**2 for c in centers) for y in points)/len(points))


def selected_branch(points, centers):
    candidates = assignments(len(points),len(centers))
    assignment = min(candidates,key=lambda a: anchor_cost(points,centers,a))
    nearest = tuple(min(range(len(centers)),key=lambda b: (y-centers[b])**2)
                    for y in points)
    return assignment,nearest


def branch_minimum(n, centers, assignment, nearest):
    k = len(centers)
    inverse = {j:a for a,j in enumerate(assignment)}
    return tuple((n*centers[inverse[j]]+k*centers[b])/(n+k)
                 if j in inverse else centers[b] for j,b in enumerate(nearest))


def branch_is_active(points, centers, assignment, nearest):
    return (anchor_cost(points,centers,assignment)
            == min(anchor_cost(points,centers,a) for a in assignments(len(points),len(centers)))
            and all((y-centers[b])**2 == min((y-c)**2 for c in centers)
                    for y,b in zip(points,nearest)))


def majorization_step(points, centers):
    assignment,nearest = selected_branch(points,centers)
    candidate = branch_minimum(len(points),centers,assignment,nearest)
    before,after = objective(points,centers),objective(candidate,centers)
    if after > before:
        raise AssertionError('exact active-quadratic minimization increased the min envelope')
    return candidate,dict(before=before,after=after,assignment=assignment,nearest=nearest)


def encode(value):
    if isinstance(value,F): return str(value)
    if isinstance(value,dict): return {key:encode(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)): return [encode(item) for item in value]
    return value


def check_geometry():
    centers=(F(-1),F(1)); n=3
    stationary=[]
    for assignment in assignments(n,len(centers)):
        for nearest in product(range(len(centers)),repeat=n):
            point=branch_minimum(n,centers,assignment,nearest)
            active=branch_is_active(point,centers,assignment,nearest)
            row=dict(assignment=assignment,nearest=nearest,branch_minimum=point,
                     active=active,objective=objective(point,centers))
            stationary.append(row)
            if active and row['objective'] != 0:
                raise AssertionError('N>K active branch stationary point has nonzero objective')
    point=(F(-1),F(-1),F(-1)); sequence=[]
    for _ in range(2):
        point,row=majorization_step(point,centers)
        sequence.append(dict(**row,point=point))
    assert objective(point,centers)==0
    good=(F(-1),F(1),F(-1))
    resting,rest=majorization_step(good,centers)
    assert resting==good and rest['after']==0
    perturbed=(F(-9,10),F(1),F(-1))
    restored,response=majorization_step(perturbed,centers)
    assert restored==good and response['after']==0

    # N=K permits a zero gradient for one active branch at a Voronoi tie,
    # even though the min envelope has a strict descent direction there.
    tie_point=(F(-1),F(0)); tie_assignment=(0,1); tie_nearest=(0,0)
    assert branch_minimum(2,centers,tie_assignment,tie_nearest)==tie_point
    assert branch_is_active(tie_point,centers,tie_assignment,tie_nearest)
    tie_perturbation=(F(-1),F(1,10))
    assert objective(tie_perturbation,centers)<objective(tie_point,centers)
    return encode(dict(status='EXACT_GEOMETRY_CHECK_PASS',
        scope='fixed distinct centers,free scalar particle coordinates;no model or optimizer',
        shared_gate_eligible=False,centers=centers,n=n,k=len(centers),
        enumerated_branches=len(stationary),active_stationary_branches=sum(row['active'] for row in stationary),
        all_active_stationary_objectives_zero=True,branches=stationary,
        wrong_subset_majorization=sequence,covered_rest=rest,
        fixed_target_perturbation_response=response,
        equal_count_caveat=dict(point=tie_point,active_branch_stationary=True,
            value=objective(tie_point,centers),descent_point=tie_perturbation,
            descent_value=objective(tie_perturbation,centers))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=check_geometry()
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='branches'}),flush=True)


if __name__=='__main__': main()
