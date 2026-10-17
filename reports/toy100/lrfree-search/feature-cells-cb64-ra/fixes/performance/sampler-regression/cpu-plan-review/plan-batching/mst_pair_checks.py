"""Bounded Prim edge order and original Torch cut checks on fixed centres."""
from copy import deepcopy
import torch
import plan_pair_common as common


def reference_edges(distance):
    k=len(distance);device=distance.device
    used=torch.zeros(k,device=device,dtype=torch.bool);used[0]=True
    nearest=distance[0].clone()
    parent=torch.zeros(k,device=device,dtype=torch.long)
    edge_parent,edge_child,edge_length=[],[],[]
    for _ in range(k-1):
        child=nearest.masked_fill(used,float('inf')).argmin()
        edge_parent.append(parent[child].clone());edge_child.append(child.clone())
        edge_length.append(nearest[child].clone())
        used[child]=True
        improve=distance[child]<nearest
        parent=torch.where(improve,child,parent)
        nearest=torch.minimum(nearest,distance[child])
    return torch.stack(edge_parent),torch.stack(edge_child),torch.stack(edge_length)


def centers_cases(data):
    for name,value in data['cases'].items():
        yield name,value['snapshot']['centers']
    fixtures=dict(
        constant3=[[0.,0.]]*3,
        constant64=[[1.,-2.]]*64,
        square_ties=[[0.,0.],[0.,1.],[1.,0.],[1.,1.]],
        duplicated_square=[[0.,0.],[0.,0.],[0.,1.],[1.,0.],[1.,1.]],
        collinear_ties=[[float(n),0.] for n in range(64)],
        separated_line=[[0.,0.],[1.,0.],[2.,0.],[20.,0.],[21.,0.],[22.,0.]],
        anisotropic=[[0.,0.],[0.,.125],[0.,.25],[16.,0.],[16.,.125],[32.,4.]],
    )
    for name,values in fixtures.items():yield name,torch.tensor(values,dtype=torch.float64)


def topology(module,centers):
    snap=module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    snap.centers=centers.clone();snap.cells=len(centers);snap.device=centers.device
    snap.work=dict(distance_cells=0,max_distance_rows=0,max_distance_columns=0,retained_array_bytes=0)
    snap._mass_topology()
    return snap


def checks(old,new,data,device):
    records=[]
    for name,source_centers in centers_cases(data):
        centers=source_centers.to(device)
        old_snap=topology(old,centers);new_snap=topology(new,centers)
        distance=old_snap._distance(centers,centers);distance.fill_diagonal_(float('inf'))
        a,b,length=reference_edges(distance)
        c,d=new._bounded_mst_edges(distance)
        edge_equal=torch.equal(a,c) and torch.equal(b,d)
        length_equal=torch.equal(length,distance[c,d])
        assert edge_equal and length_equal,name+' edge ordering/length mismatch'
        assert torch.equal(old_snap.mass_group_ids,new_snap.mass_group_ids),name+' groups changed'
        assert common.same(old_snap.mass_topology,new_snap.mass_topology),name+' threshold changed'
        # Direct distance diagnostics above add work only to the reference.
        assert new_snap.work['distance_cells']==(len(centers)**2 if len(centers)>2 else 0)
        records.append(dict(name=name,cells=len(centers),dimension=centers.shape[1],
                            edge_order_bits=True,edge_length_bits=True,cut_and_groups_exact=True,
                            topology=deepcopy(new_snap.mass_topology)))
    return records
