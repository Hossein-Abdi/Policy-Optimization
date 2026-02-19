### Procgen env
To try an environment out interactively:
```
python3 -m procgen.interactive --env-name coinrun
```

The keys are: left/right/up/down + q, w, e, a, s, d for the different (environment-dependent) actions. Your score is displayed as "episode_return" in the lower left. At the end of an episode, you can see your final "episode_return" as well as "prev_level_complete" which will be 1 if you successfully completed the level.


### Useful commands

``` bash
# interactive run on csf3 with A100
qrsh -l a100 -cwd bash

# singularity container
singularity run --writable-tmpfs --nv --no-home docker://mingfeisun/procgen:pytorch
```
