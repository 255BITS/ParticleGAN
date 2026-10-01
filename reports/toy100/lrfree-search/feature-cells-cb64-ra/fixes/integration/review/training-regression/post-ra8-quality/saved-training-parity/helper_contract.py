"""Tiny comparison controls before any RA8 numerical checkpoint is loaded."""
from copy import deepcopy
import json
import struct
import torch
from compare_saved import HERE,POLICY,EXPIRY,legacy_view,differences,digest

x=torch.tensor([0.,-0.,float('nan')]);y=x.clone()
assert not differences(x,y)
y[1]=0.;assert differences(x,y)[0]['kind']=='tensor_bytes'
assert differences({'x':torch.arange(3)},{'x':torch.arange(3)+1})[0]['path']=='$/x'
assert differences({'x':True},{'x':1})[0]['kind']=='type'
assert not differences({'x':float('nan')},{'x':float('nan')})
a=dict(completed_steps=0,models=dict(prior=dict(z=torch.zeros(1024,2))),
    birth_death=dict(backend_schema=6,settings={},snapshot_serial=0,last={},
        counters=dict(feature_distance_cells=10,projection_products=20,ordinary_moves=51)))
b=deepcopy(a);bd=b['birth_death'];bd['backend_schema']=7
bd['settings'].update(paired_average_policy=POLICY,paired_average_expiry=EXPIRY)
bd['paired_average']=dict(schema=1,policy=POLICY,rows=1024,required=973,eligible=False,step=0,snapshot=0)
bd['counters']['feature_distance_cells']=30
aa,unused=legacy_view(a,'RA7');bb,unused=legacy_view(b,'RA8')
assert not differences(aa,bb)
bad=deepcopy(b);bad['birth_death']['counters']['ordinary_moves']+=1
assert differences(aa,legacy_view(bad,'RA8')[0])[0]['path']=='$/birth_death/counters/ordinary_moves'
bad=deepcopy(b);bad['models']['prior']['z'][0,0]=1
assert differences(aa,legacy_view(bad,'RA8')[0])[0]['path']=='$/models/prior/z'
(HERE/'helper-contract.json').write_text(json.dumps(dict(status='PASS',controls=9,RA8_numerical_checkpoints_read=0,
    new_training_steps=0,new_emissions=0,cpu_only=True,quality_verdict=None),indent=2)+'\n')
print(json.dumps({'status':'PASS','controls':9,'RA8_numerical_checkpoints_read':0}))
