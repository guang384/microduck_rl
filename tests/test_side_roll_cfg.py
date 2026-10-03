"""Config-invariant and reward-sign tests for the Microduck SideRoll task.

The SideRoll is the roulade mirrored onto the body x-axis (lateral roll),
POSTURE-PRESERVING (stand → roll → stand, sit → roll → sit) and driven by a
TWO-BUTTON command: cmd = [0, roll_btn, 0], roll_btn ∈ {−1, +1} — one button
per hand; the start posture is NOT commanded (the policy infers it from
proprioception; landing targets come from the spawn bucket). These tests lock
in: the shared 61-D obs layout (hot-swap contract), the SideRollCommand
wiring, the spawn-posture-selected landing targets, the three spawn buckets
(with the stability-verified sit keyframe), the solver budgets seated contact
needs, and the reward sign conventions.
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_side_roll_env_cfg import (
    make_microduck_side_roll_env_cfg,
    MicroduckSideRollRlCfg,
    SITTING_TARGET_OVERRIDES,
    SIT_Z,
    STAND_Z,
)


def test_cfg_uses_side_roll_mdp_functions():
    cfg = make_microduck_side_roll_env_cfg()
    # main dense progress signal wired to the side-roll MDP function
    assert cfg.rewards["side_roll_progress"].func.__name__ == "side_roll_progress"
    # landing annuity gated (completion-gated standing composite)
    assert cfg.rewards["side_roll_landing_composite"].func.__name__ == "side_roll_landing_composite"


def test_command_is_a_two_button_roll():
    # The twist command must be the SideRollCommand: [0, roll_btn, 0], one
    # button per hand, NO posture flag (the start posture is inferred from
    # proprioception, not commanded).
    from mjlab_microduck.tasks.mdp import SideRollCommandCfg

    cfg = make_microduck_side_roll_env_cfg()
    assert isinstance(cfg.commands["twist"], SideRollCommandCfg)
    # Resample interval must exceed the episode so one button rules an episode.
    lo, hi = cfg.commands["twist"].resampling_time_range
    assert lo >= cfg.episode_length_s
    # no posture knob exists on the two-button command
    assert not hasattr(cfg.commands["twist"], "sit_prob")


def test_button_lives_in_the_mirror_negated_slot():
    # The mirror loss ties the two hands ONLY if the button sits in an obs slot
    # the mirror negates (vy, obs[49]). Lock the mirror table's sign pattern.
    from mjlab_microduck.tasks.symmetry import _OBS_PERM, _OBS_SIGN

    assert list(_OBS_PERM[48:51]) == [48, 49, 50]
    assert list(_OBS_SIGN[48:51]) == [1.0, -1.0, -1.0]  # vy negated — button slot


def test_landing_rewards_use_spawn_posture_targets():
    # Every landing target must be selected by the SPAWN posture (no command
    # read): the same stack returns a standing spawn to STAND and a seated
    # spawn to SIT, and mid-roll spawns pay EITHER rest (element-wise max).
    import inspect

    cfg = make_microduck_side_roll_env_cfg()
    for name in (
        "side_roll_landing_composite",
        "side_roll_height_after_roll",
        "side_roll_landing_sharp",
        "side_roll_stand_tax",
        "side_roll_rise_velocity",
    ):
        params = cfg.rewards[name].params
        assert params["stand_z"] == STAND_Z, name
        assert params["sit_z"] == SIT_Z, name
        sig = inspect.signature(cfg.rewards[name].func)
        assert "command_name" not in sig.parameters, name
    assert cfg.rewards["side_roll_landing_composite"].params["sit_overrides"] == (
        SITTING_TARGET_OVERRIDES
    )
    # the selection runs through _posture_pick on the spawn-posture tensor
    src = inspect.getsource(microduck_mdp._posture_pick)
    assert "maximum" in src
    # mid-roll spawns sample a DETERMINISTIC 50/50 stand/sit target (NOT an
    # EITHER-max): the stand-target half is the crouch→stand bootstrap —
    # an EITHER-max design neutralized it and the policy rested at sit
    # height after every roll (3000-iter eval, 2026-10-03).
    reset_src = inspect.getsource(microduck_mdp.reset_side_roll_state)
    assert "mid_target" in reset_src
    assert "full_like(u, 0.5)" not in reset_src  # old EITHER constant gone


def test_positive_side_roll_rewards_are_completion_gated():
    # The completion-gated landing functions must route through
    # _side_roll_completion_gate(require_side=True) — checked on source, since
    # building an env here is too heavy for a CPU cfg test.
    import inspect

    for fn_name in (
        "side_roll_landing_composite",
        "side_roll_upright_after_roll",
        "side_roll_height_after_roll",
        "side_roll_landing_sharp",
        "side_roll_stand_tax",
        "side_roll_rise_velocity",
    ):
        src = inspect.getsource(getattr(microduck_mdp, fn_name))
        assert "_side_roll_completion_gate" in src, fn_name
    # ...and the gate itself latches on the over-the-side pivot.
    gate_src = inspect.getsource(microduck_mdp._side_roll_completion_gate)
    assert "require_side" in gate_src


def test_penalties_are_negative_weight_and_self_negating_positive():
    cfg = make_microduck_side_roll_env_cfg()
    # progress is the only large positive "task" term
    assert cfg.rewards["side_roll_progress"].weight == 8.0
    # straightness penalties are COSTS (negative weight)
    for name in ("side_roll_sagittal", "side_roll_forward_vel", "side_roll_flatness"):
        assert cfg.rewards[name].weight < 0.0, name
    # stand_tax is SELF-NEGATING → must be positive weight (penalty sign convention)
    assert cfg.rewards["side_roll_stand_tax"].weight > 0.0


def test_always_on_upright_is_removed():
    # would oppose the flip
    cfg = make_microduck_side_roll_env_cfg()
    assert "upright" not in cfg.rewards


def test_symmetry_is_enabled_for_bilateral_double_flip():
    # A cartwheel is L/R symmetric → mirror-loss learns both handedness from one.
    assert MicroduckSideRollRlCfg.algorithm.symmetry_cfg is not None


def test_checkpointing_is_interruption_friendly():
    # The run is expected to be interrupted at arbitrary moments: save often
    # enough that an interrupt costs ≤ ~100 iterations (repo default 250 is
    # too coarse here), into a FIXED experiment dir so latest-checkpoint
    # resume (`--agent.resume True`) always finds it.
    assert MicroduckSideRollRlCfg.save_interval <= 100
    assert MicroduckSideRollRlCfg.experiment_name == "microduck_side_roll"


def test_spawn_has_buckets_with_sit_keyframe_and_crouch_start():
    cfg = make_microduck_side_roll_env_cfg()
    params = cfg.events["set_side_roll_state"].params
    assert params["sitting_joint_overrides"] == SITTING_TARGET_OVERRIDES
    total = params["standing_prob"] + params["seated_prob"] + params["midroll_prob"]
    assert 0.99 < total <= 1.01  # normalized, but keep the mix honest
    assert params["seated_prob"] > 0.0
    assert params["midroll_prob"] > 0.0
    # crouch-start reverse curriculum exists (default off, ramped by the
    # spawn-mix curriculum at iter 4600) — the crouch→stand last mile
    # needs born-at-the-frontier on-policy data
    assert params["crouch_start_prob"] == 0.0
    stages = cfg.curriculum["side_roll_spawn_mix"].params["param_stages"]
    last = {k: v for s in stages for k, v in s["params"].items()}
    assert last.get("crouch_start_prob", 0.0) > 0.0


def test_seated_contact_solver_budgets():
    # sitstand lesson: seated ground contact overflows the default solver → NaN.
    cfg = make_microduck_side_roll_env_cfg()
    assert cfg.sim.nconmax == 200
    assert cfg.sim.mujoco.iterations >= 30
    assert cfg.sim.mujoco.ls_iterations >= 50


def test_actor_observation_keeps_the_61d_slot_layout():
    cfg = make_microduck_side_roll_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    for padded in ("head_command", "body_command"):
        assert padded in terms
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6


def test_obs_layout_matches_roulade():
    from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
        make_microduck_roulade_env_cfg,
    )

    side = make_microduck_side_roll_env_cfg()
    roul = make_microduck_roulade_env_cfg()
    for grp in ("actor", "critic"):
        assert list(side.observations[grp].terms.keys()) == list(
            roul.observations[grp].terms.keys()
        ), f"layout divergent on {grp}"


def test_mdp_function_signatures():
    # The side-roll functions accept the two-button parameter set so the cfg
    # can drive them; guards against a silent API drift.
    import inspect

    for fn_name in (
        "reset_side_roll_state",
        "side_roll_progress",
        "side_roll_landing_composite",
    ):
        fn = getattr(microduck_mdp, fn_name)
        assert fn and inspect.signature(fn)
    # landing terms take BOTH rest targets (spawn-posture-selected)
    sig = inspect.signature(microduck_mdp.side_roll_landing_composite)
    for param in ("sit_overrides", "joint_indices", "stand_z", "sit_z"):
        assert param in sig.parameters, param
