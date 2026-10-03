"""Microduck SIDE-ROLL (cartwheel / lateral roll) task — roulade mirrored onto the x-axis.

Episodic, POSTURE-PRESERVING, TWO-BUTTON policy: the robot starts standing OR
seated, rolls sideways over its shoulder/trunk (rotation about the body's
FORWARD x-axis) in the COMMANDED direction, and lands back in the posture it
started from (stand → roll → stand, sit → roll → sit). The twist command is a
single ROLL BUTTON: cmd = [0, roll_btn, 0], roll_btn ∈ {−1, +1} — one button
per hand (deployment: two keys, left roll / right roll). The start posture is
NOT commanded: the policy reads it off the proprioception (standing HOME vs
the sit keyframe differ by >0.4 rad on the knees) and the landing rewards
target the SPAWN's posture. The button lives in the vy slot (obs[49]) because
the symmetry mirror table negates exactly that slot — the mirror loss then
ties the two hands together from one policy.

Design (mirrors the roulade, see microduck_roulade_env_cfg.py for the 5-run
lesson arc that produced this):
  • ONE dense progress signal — paid increments of the max-so-far cumulative
    lateral rotation (about body x), potential-based, capped at max_paid_rate,
    SIGNED by the commanded button (progress is positive in both hands).
  • Landing rewards gated on ROLL COMPLETION (lateral frontier ≥ ~260°), not a
    clock; "do nothing" earns nothing and the rest spawns can't farm them.
    Landing targets (height, joint pose) come from the spawn posture — stand
    spawns land standing, seated spawns land sitting. MID-ROLL spawns sample
    the target 50/50 stand/sit (deterministic per env): the stand-target half
    keeps the crouch→stand bootstrap alive — spawns past ~300° open the
    landing gate at birth and pay only if the duck RISES (the roulade's
    340°-spawn lesson; an earlier EITHER-max design neutralized this and the
    policy learned to rest at sit height after every roll).
  • The completion gate additionally requires the SIDE LATCH (|lateral_axis_z|
    crossed ~0.8 mid-roll) — a genuine over-the-shoulder/trunk pivot, so a
    forward tumble can't fake a cartwheel.
  • Reverse curriculum via mid-roll spawns (rotation about the x-axis, in the
    commanded direction) with the accumulator pre-set to the spawn angle.
  • Spawn buckets: STANDING at HOME / SEATED at the sit keyframe (sitstand's
    stability-verified SITTING_TARGET_OVERRIDES — seated ground contact is
    dense, hence the raised solver budgets) / MID-ROLL tucked.

DR / obs / regularisers mirror the roulade (velocity sim2real parity), with the
motion-blockers (body_ang_vel, |a_z|, torque-rate) kept near zero during
discovery and introduced late by curriculum — a cartwheel is a large
angular-velocity + impact event.

INTERRUPTION / RESUME — this task is expected to be trained in interrupted
sittings; everything needed to resume exactly is in the checkpoint (weights,
normalizer, optimizer, iteration, and common_step_counter — the curricula are
absolute functions of that counter, and all side-roll env state is reset-time,
rebuilt on the first resets). Checkpoints accumulate every `save_interval`
iterations; nothing is overwritten, so an interrupt costs at most one interval.

  GPU / HF-Jobs path (auto-resumes the LATEST checkpoint of the latest run;
  a discrete-GPU box fits 4096 envs):
    uv run train Mjlab-SideRoll-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.resume True
  Intel-GPU (SYCL) path (explicit checkpoint path; note the -sycl log dir and
  that max_iterations counts ADDITIONAL iterations after the load). MEASURED
  2026-10-02 on the Arc 130T (16 GB shared), mjlab-sycl 0.2.0 native-kernel
  suite, contended desktop: 2048 envs 4.1 s/iter (~11.9k env-steps/s), 3072
  5.7 s/iter (~12.9k), 4096 7.2 s/iter (~13.7k — fits again since the fused
  kernels shrank the USM footprint; drop to 3072 if a long session OOMs in
  alg.update with a torch XPU OutOfMemoryError):
    mjlab-sycl-train Mjlab-SideRoll-Flat-MicroDuck --num-envs 3072 \
        --save-interval 100 --run-name run1
    # after an interrupt:
    mjlab-sycl-train Mjlab-SideRoll-Flat-MicroDuck --num-envs 3072 \
        --save-interval 100 --run-name run1 \
        --checkpoint logs/Mjlab-SideRoll-Flat-MicroDuck-sycl/run1/model_0300.pt

If the newest model_*.pt fails to load (an interrupt that landed exactly
inside torch.save), delete it and resume from the previous one — every older
checkpoint is retained.
"""

import math
from copy import deepcopy

# Symmetry: left-right symmetric task → mirror-loss learns both handedness.
ENABLE_SYMMETRY = True

# ── Domain randomisation (mirrors roulade / standup for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: controlled lateral roll ~2 s + rise ~1.5 s + settle.
EPISODE_LENGTH_S = 5.0

# Empirically-measured rest heights (standup/sitstand lesson: don't guess).
STAND_Z = 0.115
SIT_Z   = 0.060

# ── SIT keyframe (servo joint index → rad) — the seated spawn AND landing pose.
# STABILITY-VERIFIED 2026-07-27 (sitstand; sweep_sit_pose2.py): settles at 3–5°
# tilt for 95–100% of noisy resets. Keep in sync with
# microduck_sitstand_env_cfg.SITTING_TARGET_OVERRIDES — if the robot or
# keyframe changes, RE-RUN THE SWEEP and verify tilt, not z.
SITTING_TARGET_OVERRIDES = {
    1:   0.0,      # left  hip_roll   (HOME -0.0873)
    2:  -0.4079,   # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3:   1.35,     # left  knee       (HOME -0.0049)
    4:   0.0,      # left  ankle      (HOME +0.4530)
    10:  0.0,      # right hip_roll   (HOME +0.0873)
    11:  0.4079,   # right hip_pitch  (HOME +0.4579)
    12: -1.35,     # right knee       (HOME +0.0049)
    13:  0.0,      # right ankle      (HOME -0.4530)
}

# ── Mid-roll spawn (reverse curriculum), mirrored onto the side ────────────
MIDROLL_PITCH_MIN   = math.radians(50.0)
MIDROLL_PITCH_MAX   = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)   # rad/s angular momentum about body x at spawn

# Tuck anchor: legs folded (crouch-anchor values from the velstand crouch
# reset) + head tucked. Same curl the roulade uses; a compact body reduces
# the roll's moment of inertia and protects the legs over the shoulder. Index
# keyed (servo layout 0-13).
TUCK_OVERRIDES = {
    2:  -1.15,  # left  hip_pitch
    3:   1.25,  # left  knee
    4:   1.05,  # left  ankle
    5:  -1.0,   # neck_pitch
    6:   1.0,   # head_pitch
    11:  1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

# Rotation thresholds (rad) for the state-based gates (same as roulade).
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO    = math.radians(180.0)
RISE_GATE_HI    = math.radians(260.0)

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_side_roll_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck lateral-roll (side roll / cartwheel) environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact — the SUPPORT GATE: the lateral-roll accumulator
    # only integrates while some robot geom touches the terrain, so a ballistic
    # flip earns no progress and never completes. NAME IS LOAD-BEARING:
    # _update_side_roll_accum reads it.
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: side-roll task set ───────────────────────────────────────────
    cfg.rewards["side_roll_progress"] = RewardTermCfg(
        func=microduck_mdp.side_roll_progress,
        weight=8.0,
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )

    cfg.rewards["side_roll_overspeed"] = RewardTermCfg(
        func=microduck_mdp.side_roll_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 7.0},
    )

    # Completion-gated rest annuity — the dominant attractor. Targets (height,
    # pose) come from the SPAWN posture (no posture command: the policy infers
    # the start posture from proprioception).
    cfg.rewards["side_roll_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.side_roll_landing_composite,
        weight=4.0,
        params={
            "sit_overrides":  SITTING_TARGET_OVERRIDES,
            "joint_indices":  _LEG_JOINTS,
            "stand_z":        STAND_Z,
            "sit_z":          SIT_Z,
            "height_std":     0.04,
            "upright_std":    0.40,
            "pose_std":       0.40,
            "gate_lo":        LANDING_GATE_LO,
            "gate_hi":        LANDING_GATE_HI,
        },
    )

    # Completion-gated bootstrap layers (gradient far from the goal).
    cfg.rewards["side_roll_upright_after_roll"] = RewardTermCfg(
        func=microduck_mdp.side_roll_upright_after_roll,
        weight=1.5,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards["side_roll_height_after_roll"] = RewardTermCfg(
        func=microduck_mdp.side_roll_height_after_roll,
        weight=1.0,
        params={
            "stand_z":      STAND_Z,
            "sit_z":        SIT_Z,
            "std":          0.04,
            "gate_lo":      LANDING_GATE_LO,
            "gate_hi":      LANDING_GATE_HI,
        },
    )

    # Sharp landing layer (tight-std upright × height product on top of the
    # broad composite) — breaks the end basin of broad stds.
    cfg.rewards["side_roll_landing_sharp"] = RewardTermCfg(
        func=microduck_mdp.side_roll_landing_sharp,
        weight=2.0,
        params={
            "stand_z":      STAND_Z,
            "sit_z":        SIT_Z,
            "height_std":   0.015,
            "upright_std":  0.3,
            "gate_lo":      LANDING_GATE_LO,
            "gate_hi":      LANDING_GATE_HI,
        },
    )

    # Completion-gated rest tax (crumple-in-a-heap must be net-negative).
    cfg.rewards["side_roll_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.side_roll_stand_tax,
        weight=5.0,
        params={
            "stand_z":      STAND_Z,
            "sit_z":        SIT_Z,
            "gate_lo":      LANDING_GATE_LO,
            "gate_hi":      LANDING_GATE_HI,
        },
    )

    # Exit-rise bootstrap: upward CoM velocity, gated to the late-roll region.
    cfg.rewards["side_roll_rise_velocity"] = RewardTermCfg(
        func=microduck_mdp.side_roll_rise_velocity,
        weight=0.75,
        params={
            "stand_z":      STAND_Z,
            "sit_z":        SIT_Z,
            "margin":       0.01,
            "gate_lo":      RISE_GATE_LO,
            "gate_hi":      RISE_GATE_HI,
        },
    )

    # Straightness — keep it a clean lateral roll (forward/vertical deviations
    # earn nothing and are taxed).
    cfg.rewards["side_roll_sagittal"] = RewardTermCfg(
        func=microduck_mdp.side_roll_sagittal_penalty,
        weight=-0.1,
    )
    cfg.rewards["side_roll_forward_vel"] = RewardTermCfg(
        func=microduck_mdp.side_roll_forward_velocity_penalty,
        weight=-0.5,
    )
    cfg.rewards["side_roll_flatness"] = RewardTermCfg(
        func=microduck_mdp.side_roll_flatness_penalty,
        weight=-0.5,
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # Arrival damper — trunk ω_xy² gated on standing height AND low tilt.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # |a_z| impact shaping — active from step 0 (SELF-NEGATING → POSITIVE weight).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Self-collision — LIGHT: a tucked roll needs body-on-body contact.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Always-on upright would oppose the flip.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / standup policies) ─────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots: zero padding for BOTH head (4) and body (6) — keep the
    # 61D obs layout parity with the velocity/standup policies.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: [0, roll_btn, 0] — the TWO-BUTTON contract. ─────────────────
    # roll_btn ∈ {−1, +1}: one button per hand (left roll / right roll). The
    # start posture is NOT commanded — the policy infers it from
    # proprioception and the landing rewards target the spawn's posture. The
    # button lives in the vy slot (obs[49]) because the symmetry mirror table
    # negates exactly that slot: the mirror loss then ties the two hands
    # together (a button in a non-negated slot would push the policy toward
    # mirror-symmetric actions that can't initiate a roll in either hand).
    # The reset event owns the episode's button (see reset_side_roll_state);
    # SideRollCommand.compute() applies it on the first step after reset —
    # mjlab resets events before the command manager, so a direct write from
    # the event would be clobbered by the command's own resample.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SideRollCommandCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Rest starts (standing at HOME / seated at the sit keyframe) + mid-roll
    # reverse-curriculum spawns (must run after reset_robot_joints — dict
    # insertion order). The event reads the command's direction and writes the
    # posture flag back so command and spawn always agree. crouch_start_prob
    # (0.0 here) is ramped to 0.35 of the standing bucket by the spawn-mix
    # curriculum at iter 4600 — the crouch→stand reverse curriculum.
    cfg.events["set_side_roll_state"] = EventTermCfg(
        func=microduck_mdp.reset_side_roll_state,
        mode="reset",
        params={
            "standing_prob":             0.35,
            "seated_prob":               0.30,
            "midroll_prob":              0.35,
            "standing_z_min":            0.11,
            "standing_z_max":            0.12,
            "standing_tilt_max":         math.radians(5.0),
            "crouch_start_prob":         0.0,
            "sitting_z_min":             0.06,   # settles to the 0.060 rest
            "sitting_z_max":             0.075,
            "sitting_tilt_max":          math.radians(8.0),
            "sitting_joint_overrides":   SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std":   0.10,
            "forward_vel_range":         (0.0, 0.0),
            "midroll_pitch_min":         MIDROLL_PITCH_MIN,
            "midroll_pitch_max":         MIDROLL_PITCH_MAX,
            "midroll_z_min":             0.05,
            "midroll_z_max":             0.10,
            "midroll_omega_range":       MIDROLL_OMEGA_RANGE,
            "tuck_overrides":            TUCK_OVERRIDES,
            "tuck_factor_range":         (0.3, 1.0),
            "joint_noise_std":           0.08,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # MuJoCo physics robustness (sitstand's seated-contact lesson): the seated
    # spawn + tucked mid-rolls put trunk, folded legs and head all in close
    # ground/self contact; the defaults overflow the contact solver → NaN →
    # nan_state terminations that punish the roll itself.
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["side_roll_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_side_roll_state",
            "param_stages": [
                {"step": 0,          "params": {"standing_prob": 0.35, "seated_prob": 0.30, "midroll_prob": 0.35}},
                {"step": 3000 * 24,  "params": {"standing_prob": 0.35, "seated_prob": 0.35, "midroll_prob": 0.30}},
                {"step": 6000 * 24,  "params": {"standing_prob": 0.40, "seated_prob": 0.40, "midroll_prob": 0.20}},
                # CROUCH-START reverse curriculum (midrun addition, 2026-10-03):
                # 35% of the standing bucket is born at the sit-keyframe crouch
                # with the landing gate pre-opened — on-policy data at the
                # "roll done, now stand up" frontier, which random exploration
                # never reaches from a post-roll crouch (z pinned at 0.062
                # through iter 4600 despite the live Gaussian gradient).
                {"step": 4600 * 24,  "params": {"crouch_start_prob": 0.35}},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 3000 * 24,  "weight": -0.4},
            ],
        },
    )

    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -0.025},
                {"step": 3500 * 24,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -5e-4},
                {"step": 3500 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.002},
                {"step": 2500 * 24,  "weight": 0.005},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckSideRollRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_side_roll",
    run_name="microduck_side_roll",
    # Interruption-friendly (repo default is 250): the run is expected to be
    # stopped at arbitrary moments, so cap the loss at ~100 iters. All older
    # checkpoints are kept — resume picks the newest intact one.
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
)