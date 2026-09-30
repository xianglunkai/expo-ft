import ml_collections
import numpy as np


def get_config():
    c = ml_collections.ConfigDict()
    c.env_type = "cobot"
    c.env_name = "cobot_screw_sorting_single"
    try:
        from client.envs.cobot_env import CobotMagicEnv
        c.env = CobotMagicEnv
    except Exception as exc:
        # GPU learner does not need the ROS env class; the IPC rollout client does.
        print(f"Not importing cobot env ({type(exc).__name__}: {exc})")

    # Trailing period matches tasks.jsonl of screw_sorting_single_sft.
    c.language_instruction = (
        "Please sort and return the silver screws in the grey box to their proper places."
    )
    c.action_space = "joint_position"
    c.gripper_action_space = "position"
    c.example_action = np.zeros((1, 7), np.float32)
    c.image_size = (180, 320)
    c.control_hz = 30
    c.auto_reset_steps = 800  # ~26s @ 30Hz
    c.edit_action_xyzg = False
    c.use_single_arm = True
    c.critic_camera_keys = ("base_0_rgb", "right_wrist_0_rgb")
    return c
