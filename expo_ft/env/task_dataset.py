"""Route offline demo loading by task env_type."""

from expo_ft.env.cobot_utils import process_cobot_dataset
from expo_ft.env.droid_utils import process_droid_dataset


def load_task_dataset(path, task_config, num_data=None):
    if task_config.env_type == "droid":
        return process_droid_dataset(path, task_config, num_data=num_data)
    if task_config.env_type == "cobot":
        return process_cobot_dataset(path, task_config, num_data=num_data)
    raise ValueError(f"Unsupported env_type: {task_config.env_type}")
