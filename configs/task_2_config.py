import pickle

import gymnasium as gym
import highway_env  # noqa: F401

# Continuous version
ENVIRONMENT = "highway-fast-v0"

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "on_road"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
        "align_to_vehicle_axes": True,
    },
    "action": {
        "type": "ContinuousAction",
        "steering_range": [-0.1, 0.1],
        "acceleration_range": [-1, 1],
    },
    "lanes_count": 8,
    "vehicles_count": 10,  # Reduced to simplify the environment
    "duration": 40,  # Shortened to allow quicker episodes
    "initial_spacing": 1.0,  # Increased to prevent immediate interactions
    "on_road_reward": 1.0,
    "action_reward": -0.1,
    "collision_reward": -1.0,
    "right_lane_reward": 0.5,
    "high_speed_reward": 1.0,  # Increased to encourage higher speeds
    "lane_change_reward": -0.05,  # Slight penalty to discourage unnecessary lane changes
    "reward_speed_range": [10, 30],  # Adjusted to match achievable speeds
    "simulation_frequency": 15,  # Increased for finer control
    "policy_frequency": 5,  # Increased to allow more frequent decisions
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "offroad_terminal": False,  # Set to False to prevent early termination
}


if __name__ == "__main__":
    # Save the config_dict to a pickle file
    with open("task2_config.pkl", "wb") as f:
        pickle.dump(config_dict, f)

    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_dict)

    # Print info
    print("Environment info:")
    print(env.spec)
    # print(env.reset())
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    print(env.observation_space.sample())
