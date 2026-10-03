"""Headless side-roll evaluation battery.

Loads a checkpoint and rolls out the four command combinations
(spawn posture {stand, sit} × roll button {+1, −1}) with deterministic
(mean) actions on CPU MuJoCo, reporting per combination:
  - completion rate: progress frontier reached ~300° (past the 260° landing
    gate) within the episode
  - side-latch rate: a genuine over-the-side pivot was crossed
  - direction correctness: net rotation signed WITH the commanded button
  - landing quality: mean final trunk tilt and height vs the posture target

Deterministic and self-contained: writes a markdown report (default
`.temp/side_roll_eval.md`), prints the same summary, always exits 0 —
built for unattended milestone checks during training.

Usage:
  python scripts/eval_side_roll.py                       # newest run1 ckpt
  python scripts/eval_side_roll.py --checkpoint path.pt --episodes 8
"""

import argparse
import dataclasses
import math
from pathlib import Path

import torch

TASK_ID = "Mjlab-SideRoll-Flat-MicroDuck"
DEFAULT_CKPT_GLOB = "logs/Mjlab-SideRoll-Flat-MicroDuck-sycl/run1/model_*.pt"
DEFAULT_OUT = ".temp/side_roll_eval.md"
STAND_Z = 0.115
SIT_Z = 0.060
COMPLETED_ANGLE = math.radians(300.0)
DIRECTION_ANGLE = math.radians(45.0)


def newest_checkpoint() -> Path:
  hits = sorted(Path(".").glob(DEFAULT_CKPT_GLOB), key=lambda p: p.stat().st_mtime)
  if not hits:
    raise SystemExit(f"no checkpoint found under {DEFAULT_CKPT_GLOB}")
  return hits[-1]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--episodes", type=int, default=8, help="envs per battery")
  parser.add_argument("--out", default=DEFAULT_OUT)
  args = parser.parse_args()
  ckpt = Path(args.checkpoint) if args.checkpoint else newest_checkpoint()
  device = "cpu"

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab_microduck.tasks.microduck_side_roll_env_cfg import (
    make_microduck_side_roll_env_cfg,
  )

  env_cfg = make_microduck_side_roll_env_cfg(play=True)
  env_cfg.scene.num_envs = args.episodes
  agent_cfg = load_rl_cfg(TASK_ID)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, dataclasses.asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  steps = int(round(env_cfg.episode_length_s / env.step_dt))
  tilt_rows: list[str] = []
  report: list[str] = [
    f"# Side-roll eval — {ckpt}",
    "",
    f"episodes/battery: {args.episodes} · deterministic mean actions · cpu",
    "",
    "| spawn | button | complete | latch | direction | wrong-way | tilt° | z |",
    "|---|---|---|---|---|---|---|---|",
  ]
  print(f"[eval] checkpoint: {ckpt}")

  for posture, stand_p, seat_p in (("stand", 1.0, 0.0), ("sit", 0.0, 1.0)):
    # force the spawn bucket via the live event term (documented mutation
    # pattern); the landing target follows the spawn posture automatically
    p = env.event_manager.get_term_cfg("set_side_roll_state").params
    p["standing_prob"], p["seated_prob"], p["midroll_prob"] = stand_p, seat_p, 0.0
    for button in (1.0, -1.0):
      obs, _ = env.reset()
      env._side_roll_spawn_cmd[:, 1] = button  # compute() applies on fresh eps
      term = env.command_manager.get_term("twist")
      term.vel_command_b[:, 1] = button

      # Best-trackers over the whole episode (immune to the timeout auto-reset
      # at the end of the loop): max_accum is the DIRECTION-SIGNED frontier, so
      # best_max >= 45° already certifies "rotated in the commanded direction"
      # (a wrong-way roll would leave it at 0); min accum catches counter-rolls.
      best_max = torch.zeros(args.episodes, device=device)
      min_accum = torch.zeros(args.episodes, device=device)
      latch_seen = torch.zeros(args.episodes, dtype=torch.bool, device=device)
      nan_hits = 0
      with torch.no_grad():
        # stop 2 steps short of the timeout boundary: the final step would
        # auto-reset and post-reset spawn state would corrupt the landing read
        for _ in range(steps - 2):
          obs, _, dones, _ = env.step(policy(obs))[:4]
          best_max = torch.maximum(best_max, env._side_roll_max)
          min_accum = torch.minimum(min_accum, env._side_roll_accum)
          latch_seen |= env._side_roll_side_latch
          nan_hits += int(torch.isnan(obs["actor"]).any(dim=-1).sum().item())

      completed = (best_max >= COMPLETED_ANGLE).float().mean().item()
      latch_rate = latch_seen.float().mean().item()
      dir_ok = (best_max >= DIRECTION_ANGLE).float().mean().item()
      wrong_way = (min_accum <= -DIRECTION_ANGLE).float().mean().item()
      quat = env.scene["robot"].data.root_link_quat_w
      tilt_deg = (
        torch.rad2deg(torch.acos((1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1, 1)))
      )
      z = env.scene["robot"].data.root_link_pos_w[:, 2]
      target_z = STAND_Z if posture == "stand" else SIT_Z
      row = (
        f"| {posture} | {button:+.0f} | {completed:.0%} | {latch_rate:.0%} "
        f"| {dir_ok:.0%} | {wrong_way:.0%} | {tilt_deg.mean().item():.1f} | {z.mean().item():.3f} |"
      )
      report.append(row)
      print(f"[eval] {row}  nan_obs={nan_hits}")

  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text("\n".join(report) + "\n", encoding="utf-8")
  print(f"[eval] report written to {args.out}")


if __name__ == "__main__":
  main()
