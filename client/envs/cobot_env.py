"""Cobot Magic single-arm env. Same RPC surface as DroidEnv for run_client.py."""
import numpy as np
from openpi_client import image_tools
from client.real_utils.detector import success_detector_manual, wait_for_episode_start
from examples.mobile_aloha_AgileX.real_env import make_real_env


class CobotMagicEnv:
    def __init__(
        self,
        language_instruction,
        image_size=(180, 320),
        control_hz=30,
        auto_reset_steps=400,
        video_dir=None,
        use_single_arm=True,
        **kwargs,
    ):
        self.language_instruction = language_instruction
        self.image_size = tuple(image_size)
        self.control_hz = control_hz
        self.auto_reset_steps = int(auto_reset_steps)
        self.video_dir = video_dir
        self._env = make_real_env(
            init_node=True,
            control_freq_hz=control_hz,
            use_single_arm=use_single_arm,  # real_env.SINGLE_ARM_NAME == "right"
        )
        self._steps = 0
        self.prev_obs = None

    def reset(self, wait_for_start=True):
        self._steps = 0
        self._env.reset()
        if wait_for_start:
            wait_for_episode_start()
        return self.get_observation()

    def _pack(self, raw):
        def _img(x):
            x = np.asarray(x)
            if x.ndim == 3 and x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))
            return image_tools.convert_to_uint8(
                image_tools.resize_with_pad(x, self.image_size[0], self.image_size[1])
            )

        imgs = raw["images"]
        qpos = np.asarray(raw["qpos"], np.float32).reshape(-1)
        if qpos.shape[0] != 7:
            raise ValueError(f"single-arm qpos must be 7D, got {qpos.shape}")
        return {
            "observation.images.cam_high": _img(imgs["cam_high"]),
            "observation.images.cam_right_wrist": _img(imgs["cam_right_wrist"]),
            "observation.state": qpos,
            "prompt": self.language_instruction,
        }

    def get_observation(self):
        raw = self._env.get_observation()
        self.prev_obs = raw
        return self._pack(raw)

    def step(self, action):
        self._steps += 1
        action = np.asarray(action, np.float64).reshape(-1)[:7]
        self._env.step(action)  # absolute joints; gripper offset is inside real_env
        return {"executed_action": action}

    def get_info_for_step(self, raw_obs=None):
        time_stop = self._steps >= self.auto_reset_steps
        manual = success_detector_manual()  # s=success, f=failure, q=discard
        success = manual == "success"
        terminate = manual == "reset" or time_stop
        done = bool(success or terminate)
        return done, success, (1.0 if success else 0.0), (0.0 if done else 1.0)

    def render(self, *args, **kwargs):
        return None
