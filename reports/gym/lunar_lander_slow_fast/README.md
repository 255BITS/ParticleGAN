# Lunar slow→fast evaluation

No Lunar rollouts have been scored in this checkout. Do not copy the 2D toy
board into this file.

The protocol is the shared control seeds: validation `391000–391019` (20) and
test `491000–491049` (50). Speed is mean steps among successful landings.
Crashes stay out of that mean. A checkpoint that lands less often, crashes
more, or leaves the map more than the #18 baseline is not a speed win.

Commands and expected paths are in
[the experiment note](../../../docs/gym-slow-fast.md).
