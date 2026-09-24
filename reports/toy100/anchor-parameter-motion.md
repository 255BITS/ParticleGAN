# Fit the functional correction relative to the current state

The post-native anchor fit keeps perfect output quality through the borrowed
hold, yet G parameter norm grows from15.75 at1200 to19.41 at2400. The largest
observed fit-Jacobian singular value rises from54.0 in the first200 updates
to93.9, then falls to88.4 in the final200. These finite observations do not
prove unbounded drift. They justify testing how the native proposal and its
corrective fit combine in parameter space.

For an affine output map with Jacobian `J`, a native displacement `d`, and a
fixed requested output change `r`, a minimum-increment correction starting
from the native proposal gives

```
net displacement = J†r + (I − J†J)d.
```

Thus the component of the native move in the kernel of `J` survives the
correction unchanged. Starting the fit from the pre-G state instead gives
`J†r`, which rests when `r=0`. This is a relative minimum-motion update, not
a pull toward zero parameters. A nonlinear Gauss–Newton sequence does not
inherit a global minimum-distance theorem from this affine identity.

The [one-state diagnostic](anchor_parameter_motion_diagnosis.py) resumes the
completed borrowed hold for exactly update2401, preserving the original
noise horizon, rates, moments and streams. It captures pre-G, native and
accepted parameters, then fits the *same* output target offline from pre-G.
This is a diagnostic, not a cold-acquisition or new-trainer pass. The
[frozen source and raw parameter archive](continuous-evidence/round6-anchor-parameter-motion/manifest.json)
binds the exact input state and output comparison.

| Displacement from pre-G | Parameter norm | Pre-Jacobian null-component norm | Null fraction |
| --- | ---: | ---: | ---: |
| Native bounded GAN proposal | .021880 | .020822 | .951650 |
| Accepted fit from native proposal | .024486 | .020841 | .851129 |
| Same target fitted from pre-G | .012888 | .00000111 | .0000858 |

Both fits converge; maximum target errors are1.74e-6 and1.43e-6. The pre-G
fit retains eight modes/HQ1 on the exact update2401 evaluation law. The
linear output-change norms of the two fitted moves are .139764 and .139781.
This establishes a local, measured source of otherwise unnecessary
parameter movement; it does not establish the cause of all earlier norm
growth or the long-run behavior of the new rule.

The separately frozen pre-start candidate changes only the fitting start.
It retains the current-batch anchor objective, native GAN proposal, actual
objective comparison, numerical budget and failed-fit rest. Saved44 and
warm200 now pass with HQ1 throughout. At1200, pre-start G norm is13.26 and
maximum observed fit-Jacobian singular value12.77, versus15.75 and54.01 with
the post-native fit. The longer neural hold is next. General distribution fidelity and scalable fitting remain
separate open requirements.
