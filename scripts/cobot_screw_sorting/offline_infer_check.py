#!/usr/bin/env python
"""Open-loop check of the RTC-SFT checkpoint on LeRobot demos.

No robot and no websocket. Loads the same RTC weights as eval_rtc_sft.sh, runs
the deployment sampler on dataset frames, and reports:

- shape and finiteness of the absolute action chunk
- whether a delay-step prefix stays pinned in normalized action space
- open-loop error against the recorded actions
- wall-clock latency of RTCLearner.sample_actions after one warmup call

Optional plots (``--plot_dir``):
- joint / gripper action curves (GT vs prediction, absolute + normalized)
- VLA inference latency vs the RTC overlap budget

The first call compiles JAX and is not included in the latency numbers.
"""

from __future__ import annotations

import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import openpi.training.sharding as openpi_sharding

from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
from expo_ft.agents.alg.rtc import load_agent
from expo_ft.agents.vla.pi05 import build_pi05
from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
from expo_ft.env.task_dataset import load_task_dataset

config_flags.DEFINE_config_file(
    "config",
    "configs/model/rtc_pi_config.py",
    "RTC agent config.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/cobot_screw_sorting.py",
    "Task config.",
    lock_config=False,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "", "LeRobot dataset root.")
flags.DEFINE_integer("num_data", 1, "Episodes to load. 0 loads all.")
flags.DEFINE_integer("num_queries", 4, "Timed inference calls per mode, after warmup.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("replan_steps", 8, "Executed window. Match deployment.")
flags.DEFINE_integer("delay", 5, "Prefix length for the delayed path. Match deployment.")
flags.DEFINE_integer("fsdp_devices", 1, "FSDP devices.")
flags.DEFINE_float(
    "prefix_atol",
    1e-3,
    "Max abs error allowed on the pinned normalized prefix.",
)
flags.DEFINE_string(
    "plot_dir",
    "./artifacts/cobot_screw_offline_check",
    "Directory for PNG plots. Empty string disables plotting.",
)
flags.DEFINE_integer(
    "plot_queries",
    2,
    "How many timed queries to draw as action curves (from the scored set).",
)

JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "grip")


def _copy_obs(frame):
    obs = frame["observations"]
    return {
        "observation.images.cam_high": obs["observation.images.cam_high"],
        "observation.images.cam_right_wrist": obs["observation.images.cam_right_wrist"],
        "observation.state": np.array(obs["observation.state"], np.float32, copy=True),
        "prompt": obs["prompt"],
    }


def _valid_starts(dataset, horizon):
    starts = []
    episode_start = 0
    for i, frame in enumerate(dataset):
        if frame["dones"] > 0.5:
            last = i - horizon
            if last >= episode_start:
                starts.extend(range(episode_start, last + 1))
            episode_start = i + 1
    return starts


def _future_actions(dataset, index, horizon):
    return np.stack([dataset[index + k]["actions"] for k in range(horizon)], axis=0).astype(np.float32)


def _normalized_actions(agent, obs, actions):
    """Dataset actions in the same normalized space the prefix uses."""
    raw = _copy_obs({"observations": obs})
    raw["actions"] = np.asarray(actions, np.float32)
    transformed = agent.actor.input_transforms(raw)
    out = np.asarray(transformed["actions"], np.float32)
    if out.shape[-1] != agent.action_dim:
        raise ValueError(f"Normalized actions ended at dim {out.shape[-1]}, expected {agent.action_dim}")
    return out


def _pad_prefix(actions_norm, model_dim):
    prefix = np.zeros((actions_norm.shape[0], model_dim), np.float32)
    prefix[:, : actions_norm.shape[-1]] = actions_norm
    return prefix


def _model_state_for_outputs(agent, obs):
    """Normalized proprio state as expected by Unnormalize + AbsoluteActions."""
    raw = _copy_obs({"observations": obs})
    raw["actions"] = np.zeros((1, int(agent.action_dim)), np.float32)
    state = np.asarray(agent.actor.input_transforms(raw)["state"], np.float32)
    model_dim = int(agent.actor.model_config.action_dim)
    if state.shape[-1] < model_dim:
        pad = np.zeros(model_dim, np.float32)
        pad[: state.shape[-1]] = state
        return pad
    return state[..., :model_dim]


def _absolute_from_model_actions(agent, obs, model_actions):
    """Unnormalize + AbsoluteActions with the *real* proprio (not dummy zeros)."""
    actions = np.asarray(model_actions, np.float32)
    model_dim = int(agent.actor.model_config.action_dim)
    if actions.ndim != 2:
        raise ValueError(f"expected (H, D) actions, got {actions.shape}")
    if actions.shape[-1] < model_dim:
        pad = np.zeros((actions.shape[0], model_dim), np.float32)
        pad[:, : actions.shape[-1]] = actions
        actions = pad
    else:
        actions = actions[:, :model_dim]
    out = agent.actor.output_transforms(
        {"state": _model_state_for_outputs(agent, obs), "actions": actions}
    )
    return np.asarray(out["actions"], np.float32)[..., : agent.action_dim]


def _pick(starts, n, rng):
    if len(starts) <= n:
        return list(starts)
    choice = rng.choice(len(starts), size=n, replace=False)
    return sorted(int(starts[i]) for i in choice)


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out[0])
    return out, (time.perf_counter() - t0) * 1000.0


def _save_plots(
    plot_dir: pathlib.Path,
    *,
    sync_ms,
    delay_ms,
    budget_ms,
    delay,
    control_hz,
    curves,
    replan_steps,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    dt_ms = 1000.0 / float(control_hz)

    # --- latency ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x0 = np.arange(1, len(sync_ms) + 1)
    x1 = np.arange(1, len(delay_ms) + 1)
    ax.plot(x0, sync_ms, "o-", label=f"delay=0  (median {_percentile(sync_ms, 50):.1f} ms)", color="#1f4e79")
    ax.plot(
        x1,
        delay_ms,
        "s-",
        label=f"delay={delay} (median {_percentile(delay_ms, 50):.1f} ms)",
        color="#c45c26",
    )
    ax.axhline(budget_ms, color="#2a7f62", ls="--", lw=1.4, label=f"overlap budget {budget_ms:.0f} ms")
    ax.set_xlabel("scored query index (after JAX warmup)")
    ax.set_ylabel("sample_actions wall time (ms)")
    ax.set_title("VLA inference latency on RTX 3090 (post-compile)")
    ax.set_xticks(sorted(set(x0.tolist() + x1.tolist())))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    lat_path = plot_dir / "vla_latency.png"
    fig.savefig(lat_path, dpi=140)
    plt.close(fig)

    # --- action curves ---
    for ci, curve in enumerate(curves):
        t = np.arange(replan_steps) * dt_ms
        gt_abs = curve["gt_abs"][:replan_steps]
        pred_abs = curve["pred_abs"][:replan_steps]
        gt_norm = curve["gt_norm"][:replan_steps]
        pred_norm = curve["pred_norm"][:replan_steps]
        dim = gt_abs.shape[-1]

        fig, axes = plt.subplots(2, dim, figsize=(2.15 * dim, 5.4), sharex=True)
        if dim == 1:
            axes = np.asarray([[axes[0]], [axes[1]]])
        for d in range(dim):
            name = JOINT_NAMES[d] if d < len(JOINT_NAMES) else f"a{d}"
            ax0 = axes[0, d]
            ax1 = axes[1, d]
            ax0.plot(t, gt_abs[:, d], "-", color="#1f4e79", lw=1.6, label="demo GT")
            ax0.plot(t, pred_abs[:, d], "--", color="#c45c26", lw=1.6, label="π0.5 pred")
            ax0.set_title(name)
            ax0.grid(True, alpha=0.3)
            if d == 0:
                ax0.set_ylabel("absolute (rad / grip)")
                ax0.legend(fontsize=8, loc="best")

            ax1.plot(t, gt_norm[:, d], "-", color="#1f4e79", lw=1.6, label="demo GT")
            ax1.plot(t, pred_norm[:, d], "--", color="#c45c26", lw=1.6, label="π0.5 pred")
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel("ms in window")
            if d == 0:
                ax1.set_ylabel("normalized")
                ax1.legend(fontsize=8, loc="best")

        fig.suptitle(
            f"open-loop action window  frame={curve['index']}  "
            f"MAE abs={curve['mae_abs']:.3f}  norm={curve['mae_norm']:.4f}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = plot_dir / f"actions_query{ci}_frame{curve['index']}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)

    # --- summary strip ---
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = ["delay=0", f"delay={delay}", "budget"]
    vals = [_percentile(sync_ms, 50), _percentile(delay_ms, 50), budget_ms]
    colors = ["#1f4e79", "#c45c26", "#2a7f62"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ms")
    ax.set_title("Median VLA latency vs RTC overlap budget")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    sum_path = plot_dir / "latency_vs_budget.png"
    fig.savefig(sum_path, dpi=140)
    plt.close(fig)

    return lat_path, sum_path


def _load_agent(dataset, config, config_task):
    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    task_description = config_task.language_instruction
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        config,
        FLAGS.seed,
        mesh,
        data_sharding,
        replicated_sharding,
        False,
        task_description,
        init_target=False,
        init_optimizer=False,
    )
    critic_camera_keys = tuple(getattr(config_task, "critic_camera_keys", CRITIC_CAMERA_KEYS))
    replay_buffer = create_replay_buffer(
        config=config,
        example_action=config_task.example_action,
        capacity=8,
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
        critic_camera_keys=critic_camera_keys,
        camera_keys=tuple(getattr(config_task, "camera_keys", ())) or None,
    )
    replay_buffer.insert_dataset(dataset[:1])
    critic_example = {
        _critic_key_to_storage(k): replay_buffer.dataset_dict[_critic_key_to_storage(k)][0][np.newaxis]
        for k in critic_camera_keys
    }
    critic_example["state"] = replay_buffer.dataset_dict["state"][0][np.newaxis]
    critic_example["actions"] = replay_buffer.dataset_dict["actions"][0][np.newaxis]
    obs_ex, state_ex, action_ex = replay_buffer.convert_to_critic_format(critic_example)
    actor.action_dim = action_ex.squeeze().shape[-1]
    actor.state_dim = state_ex.squeeze().shape[-1]
    agent_kwargs["critic_camera_keys"] = critic_camera_keys
    return load_agent(
        seed=FLAGS.seed,
        example_observation=obs_ex.squeeze(),
        example_action=action_ex.squeeze(),
        example_state=state_ex.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=False,
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        edit_action_xyzg=config_task.edit_action_xyzg,
    )


def main(_):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("offline_infer_check")
    if not FLAGS.dataset_path:
        raise ValueError("--dataset_path is required.")
    if FLAGS.delay < 0 or FLAGS.delay > FLAGS.replan_steps:
        raise ValueError(f"--delay must be in [0, replan_steps]; got {FLAGS.delay}.")

    dataset = load_task_dataset(FLAGS.dataset_path, FLAGS.config_task, FLAGS.num_data)
    agent = _load_agent(dataset, FLAGS.config, FLAGS.config_task)
    horizon = int(agent.action_horizon)
    model_dim = int(agent.actor.model_config.action_dim)
    starts = _valid_starts(dataset, horizon)
    if not starts:
        raise ValueError(f"No frame has {horizon} future actions inside one episode.")

    rng = np.random.default_rng(FLAGS.seed)
    # Two extra frames: one compile for delay=0, one for the prefix path.
    queries = _pick(starts, FLAGS.num_queries + 2, rng)
    if len(queries) < 3:
        raise ValueError(f"Need at least 3 valid starts, found {len(queries)}.")
    logger.info(
        "Loaded %d frames, %d valid starts, timing %d queries after warmup. horizon=%d delay=%d",
        len(dataset), len(starts), len(queries) - 2, horizon, FLAGS.delay,
    )

    def run_sync(obs):
        return agent.sample_actions(_copy_obs({"observations": obs}), delay=0)

    def run_delay(obs, prefix):
        return agent.sample_actions(
            _copy_obs({"observations": obs}),
            delay=FLAGS.delay,
            p1_observations=_copy_obs({"observations": obs}),
            prefix_padded=jnp.asarray(prefix),
        )

    warm_obs = dataset[queries[0]]["observations"]
    _, warm_sync = _time_call(lambda: run_sync(warm_obs))
    logger.info("warmup delay=0 %.1f ms (compile, not scored)", warm_sync)
    warm_gt = _normalized_actions(agent, warm_obs, _future_actions(dataset, queries[0], horizon))
    _, warm_delay = _time_call(lambda: run_delay(warm_obs, _pad_prefix(warm_gt[: FLAGS.delay], model_dim)))
    logger.info("warmup delay=%d %.1f ms (compile, not scored)", FLAGS.delay, warm_delay)

    sync_ms = []
    delay_ms = []
    abs_err = []
    norm_err = []
    prefix_err = []
    curves = []
    shape_ok = True
    finite_ok = True
    window = FLAGS.replan_steps

    for index in queries[2:]:
        obs = dataset[index]["observations"]
        gt = _future_actions(dataset, index, horizon)
        gt_norm = _normalized_actions(agent, obs, gt)

        (chunk, _, _), elapsed = _time_call(lambda obs=obs: run_sync(obs))
        chunk_np = np.asarray(jax.device_get(chunk))
        if chunk_np.shape != (horizon, agent.action_dim):
            shape_ok = False
            logger.error("delay=0 shape %s, expected (%d, %d)", chunk_np.shape, horizon, agent.action_dim)
        if not np.isfinite(chunk_np).all():
            finite_ok = False
        sync_ms.append(elapsed)

        inputs = agent.actor.process_raw_inputs(
            _copy_obs({"observations": obs}), agent.action_dim, agent.resize_size,
        )
        key = jax.device_put(jax.random.PRNGKey(FLAGS.seed + index), agent.actor.infer_sharding)
        pred_norm, _ = agent.actor.sample_actions(
            inputs,
            train_state=agent.actor_train_state,
            rng=key,
            train=False,
            num_samples=1,
        )
        pred_norm = np.asarray(jax.device_get(pred_norm))[0]
        pred_abs = _absolute_from_model_actions(agent, obs, pred_norm)
        mae_abs = float(np.mean(np.abs(pred_abs[:window] - gt[:window])))
        mae_norm = float(np.mean(np.abs(pred_norm[:window, : agent.action_dim] - gt_norm[:window])))
        abs_err.append(mae_abs)
        norm_err.append(mae_norm)

        if len(curves) < max(0, FLAGS.plot_queries):
            curves.append(
                dict(
                    index=int(index),
                    gt_abs=gt,
                    pred_abs=pred_abs,
                    gt_norm=gt_norm,
                    pred_norm=pred_norm[:, : agent.action_dim],
                    mae_abs=mae_abs,
                    mae_norm=mae_norm,
                )
            )

        prefix_norm = gt_norm[: FLAGS.delay]
        prefix = _pad_prefix(prefix_norm, model_dim)
        (chunk_d, _, _), elapsed_d = _time_call(lambda obs=obs, prefix=prefix: run_delay(obs, prefix))
        chunk_d = np.asarray(jax.device_get(chunk_d))
        if chunk_d.shape != (horizon - FLAGS.delay, agent.action_dim):
            shape_ok = False
            logger.error("delay=%d shape %s", FLAGS.delay, chunk_d.shape)
        if not np.isfinite(chunk_d).all():
            finite_ok = False
        delay_ms.append(elapsed_d)

        key = jax.device_put(jax.random.PRNGKey(FLAGS.seed + 1000 + index), agent.actor.infer_sharding)
        pinned = agent.actor.sample_actions_with_prefix(
            inputs,
            train_state=agent.actor_train_state,
            rng=key,
            prefix_padded=jnp.asarray(prefix),
            num_samples=1,
        )
        pinned = np.asarray(jax.device_get(pinned))[0, : FLAGS.delay]
        prefix_err.append(np.max(np.abs(pinned - prefix_norm)))

    budget_ms = FLAGS.delay / FLAGS.config_task.control_hz * 1000.0
    prefix_max = float(np.max(prefix_err)) if prefix_err else float("inf")
    prefix_ok = prefix_max <= FLAGS.prefix_atol
    logger.info("--- open-loop RTC check ---")
    logger.info("finite=%s shape=%s", finite_ok, shape_ok)
    logger.info(
        "normalized prefix pin: max_abs=%.3e atol=%.3e pass=%s",
        prefix_max, FLAGS.prefix_atol, prefix_ok,
    )
    logger.info(
        "open-loop MAE, first %d steps: absolute joints %.4f rad, normalized space %.4f",
        FLAGS.replan_steps, float(np.mean(abs_err)), float(np.mean(norm_err)),
    )
    logger.info(
        "latency delay=0 ms: median %.1f  p95 %.1f  max %.1f  (n=%d)",
        _percentile(sync_ms, 50), _percentile(sync_ms, 95), float(np.max(sync_ms)), len(sync_ms),
    )
    logger.info(
        "latency delay=%d ms: median %.1f  p95 %.1f  max %.1f  (n=%d); overlap budget %.1f ms",
        FLAGS.delay,
        _percentile(delay_ms, 50), _percentile(delay_ms, 95), float(np.max(delay_ms)), len(delay_ms),
        budget_ms,
    )
    delay_ok = _percentile(delay_ms, 50) <= budget_ms
    logger.info("median delayed infer fits inside the delay window: %s", delay_ok)
    logger.info(
        "Note: warmup 8–13 s is XLA compile, not runtime. Steady π0.5 denoising on a 3090 "
        "commonly lands near 150–200 ms; ~180 ms here is expected."
    )

    if FLAGS.plot_dir:
        plot_dir = pathlib.Path(FLAGS.plot_dir)
        lat_path, sum_path = _save_plots(
            plot_dir,
            sync_ms=sync_ms,
            delay_ms=delay_ms,
            budget_ms=budget_ms,
            delay=FLAGS.delay,
            control_hz=FLAGS.config_task.control_hz,
            curves=curves,
            replan_steps=FLAGS.replan_steps,
        )
        logger.info("wrote latency plots: %s , %s", lat_path, sum_path)
        for ci, curve in enumerate(curves):
            logger.info(
                "wrote action curve: %s",
                plot_dir / f"actions_query{ci}_frame{curve['index']}.png",
            )

    if not (finite_ok and shape_ok and prefix_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    app.run(main)
