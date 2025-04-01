import gymnasium as gym


def check_config(config_dict, expected_keys):
    assert isinstance(config_dict, dict)
    for key in expected_keys:
        assert key in config_dict


def test_task1_config():
    import task_1_config

    expected_keys = ["observation", "action", "duration"]
    check_config(task_1_config.config_dict, expected_keys)
    assert hasattr(task_1_config, "ENVIRONEMNT")
    env = gym.make(task_1_config.ENVIRONEMNT, render_mode="rgb_array")
    env.unwrapped.configure(task_1_config.config_dict)


def test_task2_config():
    import task_2_config

    expected_keys = ["observation", "action", "duration"]
    check_config(task_2_config.config_dict, expected_keys)
    assert hasattr(task_2_config, "ENVIRONMENT")
    env = gym.make(task_2_config.ENVIRONMENT, render_mode="rgb_array")
    env.unwrapped.configure(task_2_config.config_dict)


def test_task3_config():
    import task_3_config

    expected_keys = ["observation", "action", "duration"]
    check_config(task_3_config.config_dict, expected_keys)
    assert hasattr(task_3_config, "ENVIRONMENT")
    env = gym.make(task_3_config.ENVIRONMENT, render_mode="rgb_array")
    env.unwrapped.configure(task_3_config.config_dict)
