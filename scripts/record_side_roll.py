"""Record side-roll demo videos from a trained checkpoint.

Loads the policy, forces a spawn posture and roll button, rolls out one
episode with deterministic actions in a video-instrumented env, and writes
an mp4. Records what deployment will actually see: rest spawn → immediate
roll in the commanded direction → land back in the rest posture.

Usage:
  python scripts/record_side_roll.py --posture stand --button 1 --out roll_stand.mp4
  python scripts/record_side_roll.py --posture sit   --button 1 --out roll_sit.mp4
"""

import argparse
import dataclasses
from pathlib import Path

import torch

TASK_ID = "Mjlab-SideRoll-Flat-MicroDuck"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--posture", choices=("stand", "sit"), default="stand")
  parser.add_argument("--button", type=float, default=1.0, choices=(1.0, -1.0))
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  ckpt = args.checkpoint
  if ckpt is None:
    hits = sorted(
      Path(".").glob("logs/Mjlab-SideRoll-Flat-MicroDuck-sycl/run1/model_*.pt"),
      key=lambda p: p.stat().st_mtime,
    )
    ckpt = str(hits[-1])
  print(f"[record] checkpoint: {ckpt}")

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.wrappers import VideoRecorder
  from mjlab_microduck.tasks.microduck_side_roll_env_cfg import (
    make_microduck_side_roll_env_cfg,
  )

  env_cfg = make_microduck_side_roll_env_cfg(play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = 640
  env_cfg.viewer.height = 480
  # 25 cm robot: pull the camera in and lower it for a readable side view
  env_cfg.viewer.distance = 1.0
  env_cfg.viewer.elevation = -15.0
  env_cfg.viewer.azimuth = 90.0
  agent_cfg = load_rl_cfg(TASK_ID)

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
  recorder = VideoRecorder(
    env,
    video_folder=str(Path(args.out).parent),
    step_trigger=lambda step: step == 0,
    video_length=None,  # until env[0]'s episode ends = one full 5 s episode
    name_prefix=Path(args.out).stem,
    disable_logger=True,
  )
  wrapped = RslRlVecEnvWrapper(recorder)
  runner = MjlabOnPolicyRunner(
    wrapped, dataclasses.asdict(agent_cfg), device="cpu"
  )
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
  policy = runner.get_inference_policy(device="cpu")

  # force the spawn posture and pin the roll button for the whole episode
  p = env.event_manager.get_term_cfg("set_side_roll_state").params
  p["standing_prob"] = 1.0 if args.posture == "stand" else 0.0
  p["seated_prob"] = 1.0 - p["standing_prob"]
  p["midroll_prob"] = 0.0
  obs, _ = wrapped.reset()
  env._side_roll_spawn_cmd[:, 1] = args.button
  term = env.command_manager.get_term("twist")
  term.vel_command_b[:, 1] = args.button

  steps = int(round(env_cfg.episode_length_s / env.step_dt))
  with torch.no_grad():
    for _ in range(steps - 1):  # stop before the timeout-reset boundary
      obs, _, _, _ = wrapped.step(policy(obs))[:4]
  recorder.close()
  out = Path(args.out).parent / f"{Path(args.out).stem}.mp4"
  print(f"[record] wrote {out} ({steps - 1} steps, {args.posture} spawn, button {args.button:+.0f})")


if __name__ == "__main__":
  main()
