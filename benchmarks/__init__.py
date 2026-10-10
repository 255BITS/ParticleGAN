"""Repository benchmarks use the project's serial autograd scheduling policy."""
from particlegan.execution import disable_autograd_multithreading

disable_autograd_multithreading()
