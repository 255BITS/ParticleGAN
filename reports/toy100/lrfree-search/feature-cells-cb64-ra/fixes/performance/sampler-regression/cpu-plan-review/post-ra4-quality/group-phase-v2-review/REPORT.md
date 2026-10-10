# Serial GROUP-COUNT v2 lock review

PASS. The only changes from the frozen v1 wrapper are the new phase directory and `Popen(pass_fds=(lock.fileno(),))`. Reversing those two changes restores the entire v1 source bytes. All 83 original frozen input hashes still match; the owner GPU command, numerical law, inputs, GPU settings and pre/post checks are unchanged.

One private CPU subprocess fixture exercised the inherited descriptor. The dummy parent acquired a private flock, launched a dummy child with the same `pass_fds` expression and exited immediately. After that parent was reaped, a separate descriptor could not acquire the lock while the adopted dummy child was alive. Once a private file gate allowed that child to exit, the descriptor acquired the lock. The child retained its inherited descriptor until exit.

No real jobs, signals, CUDA imports or numerical tests were used. Root wrapper and preparation bytes remain unchanged. If the real wrapper exits during its child, the child now retains the cooperative serial lock. Interrupted phases may still lack a terminal PHASE-RESULT; preserve their partial evidence and logs. This review makes no GPU parity, timing or quality claim.
