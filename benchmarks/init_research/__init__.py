"""Research initializer families screened for the deterministic init study.

These install process-wide hooks and are benchmark tooling, not package API.
The shipped initializer is ``particlegan.init.deterministic_orthogonal_``.

Status: frozen replay tooling, not migrated to ``benchmarks.toy_runner``.
There is no ``ToyProblem`` here: the families work by wrapping
``torch.optim.Adam.__init__`` (and ``Tensor.normal_``/``uniform_``), so
parameters are re-initialized when an optimizer is built. The recipe-built
optimizers (``K3PGeneratorAdam``/``K3PCriticAdam``) subclass ``Adam``, so an
installed hook also fires inside ``recipe.make_*_optimizer`` and rewrites any
parameter still carrying its constructor-default init. That is init triggered
by optimizer construction, which contradicts the explicit ``particlegan.init``
API. Candidate for retirement together with the ``--init`` hooks in
``benchmarks.toy100`` and ``benchmarks.transfer_suite.toy100_compatibility``;
port any keeper to ``particlegan.init`` and call it in toy code instead.
"""
