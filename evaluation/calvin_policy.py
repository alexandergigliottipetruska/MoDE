import sys; sys.path.insert(0, "/Users/Alexander/Documents/calvin_sim/bridge")
from calvin_client import CalvinSimClient, InvalidCommandError, SimulatorError
from models.policy import policy
import torch
import numpy as np

action_policy = policy()
instruction = ["open the drawer"]

# Implement receding horizon control
def receding_horizon_control(action_chunk):
    # action_chunk is [1, 10, 7]
    # Select the first action from the generated chunk.
    first_action = action_chunk[:, 0, :]

    # Clip continuous dimensions 0–5 to [-1,1].
    continuous_dims = torch.clip(first_action[:, 0:6], min=-1.0, max=1.0).squeeze(dim=0)

    # Convert the gripper prediction to exactly -1 or +1 (choose a deterministic convention for zero).
    discrete_dims = first_action[:, 6]
    gripper_pred = torch.where(discrete_dims >= 0, 1.0, -1.0)

    a = torch.cat([continuous_dims, gripper_pred], dim=-1)

    # Convert the result to a NumPy float32 array of shape [7].
    assert a.shape == (7,)
    return a.detach().numpy().astype(np.float32)
        

with CalvinSimClient.launch(renderer="hw_noshadow") as sim:
    obs = sim.reset() 
    sim.task_begin() 

    robot_obs, scene_obs, rgb_gripper, rgb_static = obs["robot_obs"], obs["scene_obs"], obs["rgb_gripper"], obs["rgb_static"]

    # Ensure robot_obs, rgb_gripper, rgb_static in correct dtype and shape.
    robot_obs = torch.from_numpy(robot_obs).unsqueeze(0)
    rgb_gripper = torch.from_numpy(rgb_gripper).permute((2, 0, 1)).unsqueeze(0)
    rgb_static = torch.from_numpy(rgb_static).permute((2, 0, 1)).unsqueeze(0)

    for t in range(300):
        # action chunk is [1, 10, 7]
        action_chunk = action_policy(rgb_static, rgb_gripper, instruction)
        a = receding_horizon_control(action_chunk)

        # Execute action in simulation and get new observation.
        obs = sim.step(a)

        # Initial integration: receding-horizon control with an execution horizon of one.
        robot_obs, scene_obs, rgb_gripper, rgb_static = obs["robot_obs"], obs["scene_obs"], obs["rgb_gripper"], obs["rgb_static"]

        # Ensure robot_obs, rgb_gripper, rgb_static in correct dtype and shape.
        robot_obs = torch.from_numpy(robot_obs).unsqueeze(0)
        rgb_gripper = torch.from_numpy(rgb_gripper).permute((2, 0, 1)).unsqueeze(0)
        rgb_static = torch.from_numpy(rgb_static).permute((2, 0, 1)).unsqueeze(0)
