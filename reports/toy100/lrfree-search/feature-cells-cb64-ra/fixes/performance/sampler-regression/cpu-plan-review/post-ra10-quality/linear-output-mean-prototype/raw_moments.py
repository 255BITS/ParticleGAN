"""Descriptive raw-coordinate conditional moments, no oracle/scorer or draws."""
import torch
from mean_owner.mean_transport import group_means

@torch.no_grad()
def group_covariances(values,groups,total):
    means,counts=group_means(values,groups,total)
    rank=values.shape[1]
    centered=torch.zeros(total,rank,rank,dtype=torch.float64)
    # Bound temporary centered products to chunk256 * rank², never output-d².
    for start in range(0,len(values),256):
        rows=groups[start:start+256]
        delta=values[start:start+256]-means[rows]
        centered.index_add_(0,rows,delta[:,:,None]*delta[:,None,:])
    covariance=centered/(counts-1).clamp_min(1)[:,None,None]
    return means,counts,covariance

@torch.no_grad()
def raw_moment_report(snapshot,projection,real_outputs,real_features,observations,*,sigma):
    topology=snapshot._mass_topology()
    real_metric=snapshot.transform(real_features[0::2])
    real_groups=topology[snapshot._assign_metric(real_metric)[0]].cpu()
    real_values=projection.transform(real_outputs[0::2])
    means,counts,covariances=group_covariances(real_values,real_groups,snapshot.mass_groups)
    weights=counts.double()/int(counts.sum())
    clean_budget=covariances-float(sigma)**2*torch.eye(projection.rank,dtype=torch.float64)[None]
    report=dict(moment_rank=projection.rank,output_dim=projection.output_dim,selected_axes=projection.axes.tolist(),
        all_raw_coordinates_observed=projection.rank==projection.output_dim,
        real_even_counts=counts.tolist(),real_even_means=means.tolist(),real_even_centered_covariances=covariances.tolist(),
        fixed_sigma=float(sigma),real_covariance_minus_unchanged_output_noise=clean_budget.tolist(),
        negative_clean_noise_budget_groups=int((torch.linalg.eigvalsh(clean_budget).min(1).values<0).sum()),
        oracle_labels_used=False,noisy_generated_clouds_used=False,scope='Exact raw selected-coordinate moments; full raw output for dimension<=8. Descriptive only.')
    for name,observation in observations.items():
        chart=snapshot.transform(observation.features)
        groups=topology[snapshot._assign_metric(chart)[0]].cpu()
        mu,n,cov=group_covariances(observation.projected_outputs,groups,snapshot.mass_groups)
        eligible=(counts>=2)&(n>=2)
        squared=(mu-means).square().sum(1)
        report[name]=dict(counts=n.tolist(),means=mu.tolist(),centered_covariances=cov.tolist(),
            groups=groups.tolist(),defined_groups=eligible.tolist(),missing_or_small_groups=int((~eligible).sum()),
            even_weighted_squared_raw_mean_residual=float((weights*squared*eligible).sum()),
            even_weighted_centered_covariance=(weights[:,None,None]*cov*eligible[:,None,None]).sum(0).tolist(),
            even_weighted_centered_covariance_trace=float((weights*cov.diagonal(dim1=1,dim2=2).sum(1)*eligible).sum()),
            unfiltered_all_rows_mean=observation.projected_outputs.mean(0).tolist())
    return report


def report_change(before,after):
    result={}
    for name in ('FAST','EMA'):
        b,a=before[name],after[name]
        bg=torch.tensor(b['groups']);ag=torch.tensor(a['groups'])
        bc=torch.tensor(b['centered_covariances'],dtype=torch.float64);ac=torch.tensor(a['centered_covariances'],dtype=torch.float64)
        result[name]=dict(group_transitions=int((bg!=ag).sum()),
            raw_mean_objective_before=b['even_weighted_squared_raw_mean_residual'],raw_mean_objective_after=a['even_weighted_squared_raw_mean_residual'],
            raw_mean_objective_change=a['even_weighted_squared_raw_mean_residual']-b['even_weighted_squared_raw_mean_residual'],
            centered_covariance_trace_before=b['even_weighted_centered_covariance_trace'],centered_covariance_trace_after=a['even_weighted_centered_covariance_trace'],
            centered_covariance_trace_change=a['even_weighted_centered_covariance_trace']-b['even_weighted_centered_covariance_trace'],
            per_group_centered_covariance_change=(ac-bc).tolist(),
            max_absolute_centered_covariance_entry_change=float((ac-bc).abs().max()),
            missing_or_small_groups_before=b['missing_or_small_groups'],missing_or_small_groups_after=a['missing_or_small_groups'])
    return result
