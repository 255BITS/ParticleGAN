"""Read-only exact replay attribution for the cross-only near miss."""
from contextlib import contextmanager
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100 import cross_competitive_scratch as cross_module
from reports.toy100.cross_competitive_scratch import CrossCompetitiveRecorder
from reports.toy100.implicit_extra_scratch import ImplicitExtraRecorder


def cosine(a,b):
    a=a.double().flatten();b=b.double().flatten()
    denominator=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float(torch.dot(a,b)/denominator) if denominator else None


class AttributionRecorder(CrossCompetitiveRecorder):
    def __init__(self,**options):
        super().__init__(**options)
        self.attribution=[]
        self.query_losses=[]

    def step(self,optimizer,ordinary_step,closure=None):
        if not self.passthrough:
            frame=inspect.currentframe().f_back
            while frame is not None and not all(k in frame.f_locals for k in ('opt_d','opt_g','d_loss','gan')):
                frame=frame.f_back
            if frame is None:raise RuntimeError('host loss values unavailable')
            role='d' if optimizer is frame.f_locals['opt_d'] else 'g'
            value=float(frame.f_locals[role+'_loss'].detach())
            if role=='d':self.point_losses={'d':value}
            else:self.point_losses['g']=value
            del frame
        return super().step(optimizer,ordinary_step,closure)

    def _sample(self,local,z):
        generator=local['generator'];clean=getattr(generator,'model',generator)
        return clean(z) if 'means' in local else clean(local['slow'],z)

    def _observe(self,local,z,points):
        critic=local['critic'];clean=getattr(critic,'model',critic)
        if 'means' in local:
            real=local['means']
            distances=torch.cdist(points.detach(),real)
            nearest,assignment=distances.min(1)
            targets=real[assignment]
            target_labels=assignment.tolist()
            score=lambda x:clean(x)
        else:
            real=local['fast'];targets=real
            nearest=(points.detach()-real).norm(dim=1)
            target_labels=list(range(len(real)))
            score=lambda x:clean(local['slow'],x)
        input_point=points.detach().clone().requires_grad_(True)
        logits=score(input_point)
        input_gradient=torch.autograd.grad(logits.sum(),input_point,retain_graph='fast' in local)[0].detach()
        with torch.no_grad():real_logits=score(real)
        row=dict(points=points.detach().tolist(),target_labels=target_labels,
            target_error=(points.detach()-targets).tolist(),distance=nearest.tolist(),
            coordinate_mse=float((points.detach()-targets).square().mean()),
            real_logits=real_logits.tolist(),fake_logits=logits.detach().tolist(),
            advantage=float(real_logits.mean()-logits.detach().mean()),
            input_gradient=input_gradient.tolist(),
            input_gradient_norm=input_gradient.norm(dim=1).tolist())
        if 'fast' in local:
            from benchmarks.locked_shared.trajectory import PROTOCOL
            adversarial=local['gan'].g_loss(logits,real_logits.detach())
            cover=PROTOCOL['cover_weight']*torch.cdist(real,input_point).min(dim=1).values.square().mean()
            adv_gradient=torch.autograd.grad(adversarial,input_point)[0].detach()
            cover_gradient=torch.autograd.grad(cover,input_point)[0].detach()
            row.update(adversarial_gradient=adv_gradient.tolist(),cover_gradient=cover_gradient.tolist(),
                adversarial_gradient_norm=adv_gradient.norm(dim=1).tolist(),cover_gradient_norm=cover_gradient.norm(dim=1).tolist(),
                adv_cover_cosine=[cosine(a,b) for a,b in zip(adv_gradient,cover_gradient)],
                combined_gradient_norm=(adv_gradient+cover_gradient).norm(dim=1).tolist())
        return row

    def phases(self,step,opt_d,opt_g,local):
        active=self.enabled and step>=self.start_step
        if active:
            prior=local['prior'];z_before=prior.z.detach().clone()
            streams=[value for value in local.values() if isinstance(value,torch.Generator)]
            policy=local.get('noise_policy')
            if policy is not None:streams.extend(value for name in ('input_stream','output_stream')
                if isinstance((value:=getattr(policy,name,None)),torch.Generator))
            streams=list({id(s):s for s in streams}.values())
            rng=self._rng(streams)
            before_parameters={p:p.detach().clone() for opt in (opt_d,opt_g) for group in opt.param_groups for p in group['params']}
            with torch.no_grad():before=self._sample(local,z_before)
            base=self._observe(local,z_before,before)
            if not all(torch.equal(a,b) for a,b in zip(rng,self._rng(streams))):
                raise RuntimeError('base attribution changed a training RNG stream')
        yield from super().phases(step,opt_d,opt_g,local)
        if not active:return
        rng=self._rng(streams)
        with torch.no_grad():
            after_network=self._sample(local,z_before)
            after=self._sample(local,prior.z)
        final=self._observe(local,prior.z,after)
        generator_move=after_network-before
        prior_move=after-after_network
        total_move=after-before
        old_error=torch.tensor(base['target_error'])
        old_input_gradient=torch.tensor(base['input_gradient'])
        offsets=[];cursor=0
        for opt in self.optimizers:
            for group in opt.param_groups:
                n=sum(p.numel() for p in group['params'])
                role='d' if opt is opt_d else 'prior' if group.get('_comparison_prior') else 'g'
                move=torch.cat([(p.detach()-before_parameters[p]).flatten().double() for p in group['params']])
                metric_move=move/self.root_metric[cursor:cursor+n]
                field=self.solve_q0[cursor:cursor+n]
                offsets.append(dict(role=role,move_norm=float(move.norm()),metric_move_norm=float(metric_move.norm()),
                    field_metric_norm=float(field.norm()),move_against_field_cosine=cosine(metric_move,-field)))
                cursor+=n
        row=dict(completed_step=step+1 if 'means' in local else step,base=base,final=final,
            g_move=generator_move.tolist(),prior_move=prior_move.tolist(),total_move=total_move.tolist(),
            mean_g_output_move=float(generator_move.norm(dim=1).mean()),
            mean_prior_output_move=float(prior_move.norm(dim=1).mean()),
            move_toward_targets_cosine=cosine(total_move,-old_error),
            move_with_critic_gradient_cosine=cosine(total_move,old_input_gradient),
            target_error_work=(total_move*old_error).sum(dim=1).tolist(),
            label_changes=sum(a!=b for a,b in zip(base['target_labels'],final['target_labels'])),
            alpha=self.last_scale,roles=offsets)
        self.attribution.append(row)
        if not all(torch.equal(a,b) for a,b in zip(rng,self._rng(streams))):
            raise RuntimeError('final attribution changed a training RNG stream')

    def receipt(self):
        value=super().receipt()
        value['attribution']=self.attribution
        value['query_losses']=self.query_losses
        value['attribution_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['warm','cold'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    declaration=dict(task=args.task,observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method='read-only clean support/logit/input-gradient observation; all training RNG streams checked unchanged',
        original_head='63e0788',shared_gate_eligible=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.attribution-declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    sys.argv=[sys.argv[0],'--output',str(args.output)]
    if args.task=='warm':
        from reports.toy100 import cross_competitive_warm_probe as driver
    else:
        from reports.toy100 import cross_competitive_probe as driver
    ordinary_evaluate=ImplicitExtraRecorder._evaluate
    def loss_observer(self,u,kind):
        value=yield from ordinary_evaluate(self,u,kind)
        if isinstance(self,AttributionRecorder):
            self.query_losses.append(dict(outer_step=self.outer_steps+1,kind=kind,**self.point_losses))
        return value
    with patch.object(cross_module,'CrossCompetitiveRecorder',AttributionRecorder),patch.object(ImplicitExtraRecorder,'_evaluate',loss_observer):driver.main()
    relative='cross_only.json' if args.task=='warm' else 'trajectory.json'
    reference=ROOT/'artifacts'/('continuous-cross-competitive-warm' if args.task=='warm' else 'continuous-cross-competitive-cold')/relative
    observed=json.loads((args.output/relative).read_text());expected=json.loads(reference.read_text())
    if args.task=='warm':
        parity=(observed['final_state_sha256']==expected['final_state_sha256'] and observed['diagnostic']==expected['diagnostic'])
    else:
        def metrics(value):return [{key:item for key,item in point.items() if key!='seconds'} for point in value['result']['observations']]
        parity=observed['result']['live']==expected['result']['live'] and metrics(observed)==metrics(expected)
    if not parity:raise RuntimeError('read-only attribution changed the reference replay')
    (args.output/'attribution-parity.json').write_text(json.dumps(dict(exact_reference_parity=True,reference=str(reference)),indent=2)+'\n')
    print(json.dumps(dict(event='ATTRIBUTION_PARITY',exact_reference_parity=True)),flush=True)


if __name__=='__main__':main()
