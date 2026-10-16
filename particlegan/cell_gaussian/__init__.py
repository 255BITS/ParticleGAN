"""Training-only conditional moments of the public served MoG mixture."""
from copy import deepcopy
import torch
from .cell_loss import ConsumedCellGaussianKL
from .legacy_cells import assigned_loss

MODES = ('per_kernel', 'aggregate_forward_kl', 'aggregate_reverse_kl',
         'aggregate_bures', 'aggregate_moment')

def aggregate_loss(mean, covariance, reference, assignment, mode, shape_weighting="mass",return_terms=False):
    """Quota-weighted moments include both within- and between-kernel variance."""
    cells, dimension = reference['means'].shape
    counts = torch.bincount(assignment, minlength=cells)
    if not bool((counts > 0).all()):
        raise ValueError('Empty quota cell')
    center = mean.new_zeros(cells, dimension).index_add(0, assignment, mean) / counts[:, None]
    residual = mean - center[assignment]
    total = covariance + residual[:, :, None] * residual[:, None, :]
    mixture_cov = mean.new_zeros(cells, dimension, dimension).index_add(0, assignment, total) / counts[:, None, None]
    identity = torch.eye(dimension, device=mean.device, dtype=mean.dtype)
    mixture_cov = mixture_cov + reference['jitter'] * identity
    target_cov = reference['covariance']
    delta = center - reference['means']
    sign, logdet = torch.linalg.slogdet(mixture_cov)
    if not bool((sign.detach() > 0).all()):
        raise ValueError('Nonpositive generated mixture covariance')
    if mode == 'aggregate_reverse_kl':
        inverse = torch.linalg.inv(mixture_cov)
        placement = torch.einsum('ki,kij,kj->k', delta, inverse, delta) / dimension
        shape = (torch.einsum('kij,kji->k', inverse, target_cov) + logdet - reference['logdet'] - dimension) / dimension
    else:
        placement = torch.einsum('ki,kij,kj->k', delta, reference['inverse'], delta) / dimension
        if mode == 'aggregate_forward_kl':
            shape = (torch.einsum('kij,kji->k', reference['inverse'], mixture_cov) - logdet + reference['logdet'] - dimension) / dimension
        else:
            values, vectors = torch.linalg.eigh(target_cov)
            inverse_root = (vectors * values.rsqrt()[:, None, :]) @ vectors.transpose(-2, -1)
            whitened = inverse_root @ mixture_cov @ inverse_root
            whitened = (whitened + whitened.transpose(-2, -1)) / 2
            if mode == 'aggregate_bures':
                # Only eigenvalues are needed; repeated-eigenvector derivatives are avoided.
                eigenvalues = torch.linalg.eigvalsh(whitened)
                if not bool((eigenvalues.detach() > 0).all()):
                    raise ValueError('Nonpositive whitened covariance')
                shape = (whitened.diagonal(dim1=-2, dim2=-1).sum(-1) + dimension - 2 * eigenvalues.sqrt().sum(-1)) / dimension
            elif mode == 'aggregate_moment':
                shape = .5 * (whitened - identity).square().sum((-2, -1)) / dimension
            else:
                raise ValueError('Unknown aggregate moment geometry')
    weights = counts.to(mean.dtype) / len(mean)
    shape_weights = weights if shape_weighting == "mass" else torch.ones_like(weights) / cells
    parts = dict(placement=(weights * placement).sum(), shape=shape.max() if shape_weighting=='worst' else (shape_weights * shape).sum())
    if return_terms:return parts['placement']+parts['shape'],parts,weights*(placement+shape)
    return parts['placement'] + parts['shape'], parts

class CellObjective(ConsumedCellGaussianKL):
    def __init__(self, mode, reference_policy="cumulative",quota_policy="largest_remainder",shape_weighting="mass",assignment_refinement="none",transport_policy="integer",distribution_shape="none",covariance_memory="unchanged",gradient_projection=False):
        if mode not in MODES:
            raise ValueError('Unknown cell Gaussian objective')
        super().__init__()
        self.mode = mode
        self.reference_policy=reference_policy
        self.quota_policy=quota_policy
        self.shape_weighting=shape_weighting
        self.assignment_refinement=assignment_refinement
        self.transport_policy=transport_policy
        self.distribution_shape=distribution_shape
        self.covariance_memory=covariance_memory
        self.gradient_projection=gradient_projection
        self.previous_assignment=None
        self.gradient_terms=None
        self.assignment_diagnostic={}
        self.memory=None
        if reference_policy!="cumulative":
            from .memory import ReferenceMemory
            self.memory=ReferenceMemory(reference_policy,quota_policy)

    def assigned_loss(self, mean, covariance, reference):
        if self.transport_policy == "fractional_exact":
            from .transport import fractional_loss
            value, parts, assignment, self.assignment_diagnostic = fractional_loss(mean, covariance, reference)
            return value, parts, assignment
        value, parts, assignment = assigned_loss(mean, covariance, reference)
        if self.assignment_refinement == 'best_single_swap':
            from .refinement import best_single_swap
            assignment, self.assignment_diagnostic = best_single_swap(mean, covariance, reference, assignment)
        if self.assignment_refinement=='minimum_switch':
            if self.previous_assignment is not None:
                from .legacy_cells import moment_cost
                from .portfolio import minimum_switch
                assignment,self.assignment_diagnostic=minimum_switch(moment_cost(mean,covariance,reference)[0],reference['slots'],self.previous_assignment)
            self.previous_assignment=assignment.detach().clone()
        if self.mode != 'per_kernel':
            if self.gradient_projection:
                value,parts,self.gradient_terms=aggregate_loss(mean,covariance,reference,assignment,self.mode,self.shape_weighting,True)
            else:
                value, parts = aggregate_loss(mean, covariance, reference, assignment, self.mode, self.shape_weighting)
        return value, parts, assignment

    def __call__(self, fake, real, trainer):
        self.gradient_terms=None
        if type(trainer.prior).__name__ != 'MoGParticlePrior' or trainer.prior.standardize or float(trainer.prior.sigma) <= 0:
            raise ValueError('Positive-width unstandardized MoG required')
        if self.memory is None:return super().__call__(fake, real, trainer)
        from .legacy_cells import cubature_moments
        from .parent_control import log_mass_deviance_loss
        self.bank.append(real.detach().clone());self.steps+=1
        refit=self.memory.consume(real)
        self.statistics=self.memory.statistics
        self.reference_row_usages=self.memory.seen_rows
        if self.steps<8:return log_mass_deviance_loss(fake,real)
        old_allocations=None if self.reference is None else self.reference['allocations'].clone()
        old_covariance=None if self.reference is None else self.reference["covariance"].clone()
        if refit:
            self.reference=self.memory.reference(trainer.prior.num_particles,real.dtype);self.fit_step=self.steps
            self.fit_history.append(dict(step=self.steps,reference_rows=self.reference["reference_rows"],policy=self.reference_policy))
        if refit and self.covariance_memory=="slow32" and old_covariance is not None:
            covariance=(31/32)*old_covariance+(1/32)*self.reference["covariance"]
            self.reference.update(covariance=covariance,inverse=torch.linalg.inv(covariance),logdet=torch.linalg.slogdet(covariance)[1])
        mean,cov,extra=cubature_moments(trainer.G,trainer.prior)
        value,parts,assignment=self.assigned_loss(mean,cov,self.reference)
        if self.distribution_shape!="none":
            from .distribution_shape import loss
            shape,rows,diagnostic=loss(trainer.G,trainer.prior,self.reference,assignment,self.bank,self.distribution_shape)
            value=value+shape;parts["conditional_shape"]=shape;extra+=rows
            self.assignment_diagnostic.update(diagnostic)
        self.forward_rows+=extra
        self.last=dict(stage="DATA_MEMORY_CELL_MOMENTS",step=self.steps,cells=len(self.reference["means"]),extra_forward_rows=extra)
        allocations=self.reference['allocations']
        expected=self.reference['counts']/self.reference['counts'].sum()*len(self.reference['slots'])
        self.last.update(quota_policy=self.quota_policy,allocations=allocations.detach().cpu().tolist(),
            expected_allocations=expected.detach().cpu().tolist(),assignment=assignment.detach().cpu().tolist(),
            minimum_quota_moves=0 if old_allocations is None else int((allocations-old_allocations).abs().sum())//2,
            cell_loss=float(value.detach()))
        self.last.update(shape_weighting=self.shape_weighting,assignment_refinement=self.assignment_refinement,transport_policy=self.transport_policy,distribution_shape=self.distribution_shape,**self.assignment_diagnostic)
        return value,{k:v.detach() for k,v in parts.items()}

    def state_dict(self):
        return dict(mode=self.mode,reference_policy=self.reference_policy,quota_policy=self.quota_policy,shape_weighting=self.shape_weighting,assignment_refinement=self.assignment_refinement,transport_policy=self.transport_policy,distribution_shape=self.distribution_shape,covariance_memory=self.covariance_memory,gradient_projection=self.gradient_projection,previous_assignment=self.previous_assignment,memory=None if self.memory is None else self.memory.state_dict(),state=super().state_dict())

    def validate_state(self, value, steps):
        if not isinstance(value, dict) or set(value) != {'mode','reference_policy','quota_policy','shape_weighting','assignment_refinement','transport_policy','distribution_shape','covariance_memory','gradient_projection','previous_assignment','memory','state'} or value['mode'] != self.mode or value['reference_policy']!=self.reference_policy or value['quota_policy']!=self.quota_policy or value['shape_weighting']!=self.shape_weighting or value['assignment_refinement']!=self.assignment_refinement or value['transport_policy']!=self.transport_policy or value['distribution_shape']!=self.distribution_shape or value['covariance_memory']!=self.covariance_memory or value['gradient_projection']!=self.gradient_projection:
            raise ValueError('Cell objective checkpoint mode mismatch')
        state = value['state']
        if not isinstance(state, dict) or set(state) != set(super().state_dict()) or state['steps'] != steps:
            raise ValueError('Cell objective checkpoint steps/schema mismatch')
        def finite(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point() and not bool(torch.isfinite(x).all()):
                raise ValueError('Nonfinite cell checkpoint')
            if isinstance(x, dict):
                for v in x.values(): finite(v)
            if isinstance(x, (list, tuple)):
                for v in x: finite(v)
        finite(value)
        previous=value['previous_assignment']
        if self.assignment_refinement=='minimum_switch' and steps>=8:
            ref=state['reference']
            if not isinstance(previous,torch.Tensor) or previous.dtype!=torch.long or previous.shape!=ref['slots'].shape or bool((previous<0).any()) or not torch.equal(torch.bincount(previous,minlength=len(ref['allocations'])),ref['allocations']):
                raise ValueError('Previous assignment capacity mismatch')
        elif previous is not None:raise ValueError('Unexpected previous assignment')
        if len(state['bank']) != min(steps, 8):
            raise ValueError('Cell checkpoint bank mismatch')
        if self.memory is None:
            if value['memory'] is not None:raise ValueError('Unexpected reference memory')
        else:
            memory=value['memory']
            if not isinstance(memory,dict) or memory['policy']!=self.reference_policy or memory['steps']!=steps or len(memory['bank'])!=min(steps,8):
                raise ValueError('Reference memory checkpoint mismatch')
            if memory['seen_rows']!=state['reference_row_usages']:
                raise ValueError('Reference memory observed row count mismatch')
            def equal(a,b):
                if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
                if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
                if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                return a==b
            if not equal(memory['statistics'],state['statistics']) or not equal(memory['bank'],state['bank']):
                raise ValueError('Reference memory duplicate state mismatch')
            if self.reference_policy=='exponential8':
                previous=memory.get('previous_allocations')
                if steps<8:
                    if previous is not None:raise ValueError('Premature quota state')
                elif not isinstance(previous,torch.Tensor) or not torch.equal(previous,state['reference']['allocations']):
                    raise ValueError('Reference allocation state mismatch')
                if steps>=8:
                    reference=state['reference'];slots=reference['slots']
                    if previous.ndim!=1 or previous.dtype!=torch.long or int(previous.sum())!=len(slots) or not bool((previous>0).all()):
                        raise ValueError('Invalid exact capacity checkpoint')
                    if not torch.equal(torch.bincount(slots,minlength=len(previous)),previous):
                        raise ValueError('Capacity slots checkpoint mismatch')
                    if self.quota_policy=='minimum_change':
                        expected=reference['counts']/reference['counts'].sum()*len(slots)
                        if not bool(((previous>=expected.floor())&(previous<=expected.ceil())).all()):
                            raise ValueError('Quota checkpoint violates current mass bounds')
        clone = CellObjective(self.mode,self.reference_policy,self.quota_policy,self.shape_weighting,self.assignment_refinement,self.transport_policy,self.distribution_shape,self.covariance_memory,self.gradient_projection)
        clone.load_state_dict(deepcopy(value))

    def load_state_dict(self, value):
        if value['mode'] != self.mode or value['quota_policy']!=self.quota_policy or value['reference_policy']!=self.reference_policy or value['shape_weighting']!=self.shape_weighting or value['assignment_refinement']!=self.assignment_refinement or value['transport_policy']!=self.transport_policy or value['distribution_shape']!=self.distribution_shape or value['covariance_memory']!=self.covariance_memory or value['gradient_projection']!=self.gradient_projection:
            raise ValueError('Cell objective mode mismatch')
        previous=value['previous_assignment']
        if previous is not None and (not isinstance(previous,torch.Tensor) or previous.dtype!=torch.long or previous.ndim!=1 or bool((previous<0).any())):
            raise ValueError('Invalid previous assignment checkpoint')
        self.previous_assignment=None if previous is None else previous.clone()
        super().load_state_dict(value['state'])
        if self.memory is not None:
            self.memory.load_state_dict(value['memory'])
            self.statistics=self.memory.statistics
