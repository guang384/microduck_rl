# Side-roll policy — how it was trained

Two-button lateral cartwheel: stand/sit → roll over the shoulder in the commanded
direction → land back in the START posture (no posture command; the policy reads it
off proprioception). Task `Mjlab-SideRoll-Flat-MicroDuck`, cfg
`src/mjlab_microduck/tasks/microduck_side_roll_env_cfg.py` (the module docstring holds
the task design and its lesson arc).

**Current policy: local checkpoint `model_7398.pt`**
(`logs/Mjlab-SideRoll-Flat-MicroDuck-sycl/run1/`). No wandb run — trained locally
with tensorboard. Published stand-alone (public):
[guang384/microduck-side-roll-right](https://huggingface.co/guang384/microduck-side-roll-right) v1.2
and [guang384/microduck-side-roll-left](https://huggingface.co/guang384/microduck-side-roll-left) v1.1
(`duration_s` 2.0; direction via `command.idle` [0,±1,0]).

Command contract: the twist slot is **two buttons** `[0, roll_btn, 0]`, btn ∈ {−1,+1}
in the vy slot — deliberately the slot the symmetry mirror negates, so the mirror loss
ties both hands to one policy. The start posture is not commanded.

## Stage 1 — base run, 0 → 6000 iters (from scratch)

| | |
|---|---|
| code | branch `feat/sycl-training-docs` (pre-archive working tree, merge `f2453da`) |
| host | this laptop, Intel Arc 130T iGPU via **mjlab-sycl** (physics on `sycl`, PPO on `xpu`) |
| envs | 3072 (4096 XPU-OOMs — seated-roll contact budgets; see capacity below) |
| command | see "How it was trained on Intel" below |

```bash
mjlab-sycl-train Mjlab-SideRoll-Flat-MicroDuck --num-envs 3072 \
  --save-interval 100 --max-iterations 6000 --run-name run1
```

Roll mechanics learned by ~iter 1000 (progress rewards accelerating), roll metrics
near-perfect by ~4500. **Three mid-run interventions** (each = stop, patch, warm
resume from the newest checkpoint — curricula re-derive from `common_step_counter`,
so resumes are lossless):

1. **iter ~3100** — mid-roll landing target was EITHER-max (max of the stand/sit
   scores), which neutralized the crouch→stand pressure: the policy learned to rest
   at sit height after every roll. Fix: mid-roll target = 50/50 deterministic
   stand/sit sample (restores roulade's gate-open-at-birth bootstrap).
2. **iter ~4600** — added the crouch-start reverse curriculum: 35 % of the standing
   bucket born at the sit-keyframe crouch with the landing gate pre-opened.
3. **iter ~6000** — crouch-start upgraded to multi-depth (z 0.06–0.11, joint pose
   lerped by depth) after single-depth starts failed to produce the rise in 1400
   iterations.

## Stage 2 — extension to 7398 (multi-depth crouch-start)

| | |
|---|---|
| code | branch `side-roll`, commit `8ad405a` |
| start | `model_5999.pt` resumed |
| checkpoint used | **`model_7398.pt`** (final) |

Final evaluation (`scripts/eval_side_roll.py`, 12 episodes per combination,
deterministic actions): complete 100 % on all four combinations (stand/sit ×
left/right), latch 100 %, direction 100 %, wrong-way 0 %, landing tilt ~1.2°.
**Sit-preservation perfect** (sit → roll → sit, trunk z 0.063 ≈ target).
**Known gap:** stand-start rolls land at crouch height and the policy never
learned to stand back up (see gap below).

## How it was trained on Intel (mjlab-sycl)

```bash
python -m mjlab_sycl install            # warp SYCL overlay + self-check (re-run after every uv sync)
mjlab-sycl-train Mjlab-SideRoll-Flat-MicroDuck --num-envs 3072 \
  --save-interval 100 --max-iterations N --run-name run1
# resume after an interrupt / OOM:
mjlab-sycl-train Mjlab-SideRoll-Flat-MicroDuck --num-envs 3072 \
  --save-interval 100 --max-iterations N --run-name run1 \
  --checkpoint logs/Mjlab-SideRoll-Flat-MicroDuck-sycl/run1/model_XXXX.pt
# preflight + capacity (measured, Arc 130T, seated-roll budgets nconmax 200 / solver 30/50):
mjlab-sycl-check
mjlab-sycl-bench --device sycl --task Mjlab-SideRoll-Flat-MicroDuck --num-envs 3072 --iters 5
#   4096 envs: XPU OOM in alg.update   3072: ~5.7 s/iter   2048: ~4.1 s/iter
```

Training ran ~14 h wall across three interrupted sessions (7398 iters, zero NaN).
Full footgun list: the mjlab-sycl package's README-SYCL-TRAINING.md; the resume
commands are also in the cfg module docstring.

## Known gap — the stand-up last mile

Stand-start rolls land at crouch height (trunk z 0.062) and stay there. Root cause
(settle-probe, measured): the intermediate crouch depths on the way back to standing
are **not equilibria** — 88–100 % collapse back to the sit rest within 2 s — so the
reverse-curriculum data consists entirely of collapse trajectories; the policy never
observes a rising one. The reward gradient exists (height Gaussian slope + stand tax)
but is unreachable by exploration. Fix candidates: slewed-setpoint landing rewards
(sitstand's route — track a slowly rising height target) or equilibrium-verified
intermediate poses. The sit-start loop needs none of this and is complete.

## Evaluation

```bash
./.venv/Scripts/python.exe scripts/eval_side_roll.py \
  --checkpoint logs/Mjlab-SideRoll-Flat-MicroDuck-sycl/run1/model_7398.pt --episodes 12
# videos: ./scripts/record_side_roll.py --posture stand --button 1 --out roll_stand.mp4
```
