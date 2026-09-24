import torch

from reports.toy100.forward_kl_chunked import donor_values, global_donor
from reports.toy100.forward_kl_free_filter import cross_entropy, quadrature


def test_chunked_replacements_equal_exhaustive_full_objective_with_dominant_atoms():
    real=torch.tensor([[-3.,0.],[-2.8,.07],[2.9,.1],[3.1,-.04]],dtype=torch.float64)
    points=torch.tensor([[-3.1,0.],[-2.9,.03],[-2.7,.1]],dtype=torch.float64)
    locations,weights=quadrature(real,.03,5)
    variance=.03**2+.029**2
    expected=torch.empty((len(points),len(real)),dtype=torch.float64)
    for donor in range(len(points)):
        for candidate in range(len(real)):
            proposal=points.clone()
            proposal[donor]=real[candidate]
            expected[donor,candidate]=cross_entropy(locations,weights,proposal,variance)
    # Small and nondividing chunk boundaries exercise sum order and the tail.
    for size in (1,17,2048):
        actual,accounting=donor_values(real,points,locations,weights,variance,chunk_rows=size)
        torch.testing.assert_close(actual,expected,rtol=2e-14,atol=1e-10)
        # The symmetric right-hand candidates tie mathematically; reduction
        # order can exchange their last bits without changing the minimizer.
        selected_full=expected.flatten()[actual.argmin()]
        torch.testing.assert_close(selected_full,expected.min(),rtol=0,atol=1e-10)
        assert accounting['maximum_candidate_matrix_elements']<=len(real)*min(size,len(locations))


def test_global_donor_repairs_unrepresented_cloud_without_rng_or_input_mutation():
    real=torch.tensor([[-2.,0.],[-1.9,.1],[2.,0.],[2.1,-.1]],dtype=torch.float64)
    points=torch.tensor([[-2.,0.],[-2.1,.1],[-1.9,-.1]],dtype=torch.float64)
    locations,weights=quadrature(real,.04,5)
    before=points.clone()
    rng=torch.random.get_rng_state().clone()
    result,rows=global_donor(real,points,locations,weights,.003,limit=3,chunk_rows=31)
    assert rows and all(r['after']<r['before'] for r in rows)
    assert float((result[:,0]>0).sum())>=1
    assert torch.equal(points,before)
    assert torch.equal(torch.random.get_rng_state(),rng)
