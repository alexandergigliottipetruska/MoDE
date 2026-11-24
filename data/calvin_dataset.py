import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

dataset_dir = Path("/Users/Alexander/Documents/calvin_data/calvin_debug_dataset/training")

class CALVINDataset(Dataset):
    def __init__(self, root_dir, horizon=10, transform=None):
        """
        Arguments:
            root_dir (string): directory with all the data
            transform (callable, optional): optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.horizon = horizon

        # tells you which instruction applies to which frame numbers
        file_path = self.root_dir / "lang_annotations/auto_lang_ann.npy"

        # Load language annotations
        lang_ann =  np.load(file_path, allow_pickle=True).item()

        self.language_instructions = lang_ann["language"]["ann"]
        instruction_idex = list(range(0, len(self.language_instructions)))

        self.frame_annotations = lang_ann["info"]["indx"]

        # A list of tuples of the form (frame_number, language_instruction_num)
        self.valid_frame_annotations = []

        for i in range(len(instruction_idex)):
            for j in range(self.frame_annotations[i][0], self.frame_annotations[i][1]+1):
                # Get start and end frames
                start_frame = self.frame_annotations[i][0]
                end_frame = self.frame_annotations[i][-1]
                if j <= end_frame - 9:
                    self.valid_frame_annotations.append((j, i))


    def __len__(self):
        return len(self.valid_frame_annotations)

    def __getitem__(self, idx):

        frame, annotation_idx = self.valid_frame_annotations[idx]
        current_instruction_frame_range = self.frame_annotations[annotation_idx]
        start_frame = current_instruction_frame_range[0]
        end_frame = current_instruction_frame_range[-1]

        # store initial frame of action window
        t = frame
        assert start_frame <= t

        # Load episode_framenumber.npz
        frame_path = self.root_dir / f"episode_0{frame}.npz"
        with np.load(frame_path, allow_pickle=True) as data:
            # Extract its images, robot state, and relative action
            robot_obs = data["robot_obs"]
            rgb_static = data["rgb_static"]
            rgb_gripper = data["rgb_gripper"]

            # Retrieve language instruction
            language_instruction = self.language_instructions[annotation_idx]

        # Preprocessing
        # Convert [H, W, C] to [C, H, W] and convert to correct data types
        rgb_static = torch.from_numpy(np.permute_dims(rgb_static, (2, 0, 1)))
        rgb_gripper = torch.from_numpy(np.permute_dims(rgb_gripper, (2, 0, 1)))
        robot_obs = torch.from_numpy(robot_obs).to(torch.float32)

        action_chunk = []
        assert t + 9 <= end_frame
        # Extract an action chunk of relative actions
        for i in range(10):
            with np.load(self.root_dir / f"episode_0{frame}.npz", allow_pickle=True) as data:
                rel_actions = torch.from_numpy(data["rel_actions"]).to(torch.float32)
                action_chunk.append(rel_actions)

            frame += 1

        action_chunk = torch.stack(action_chunk, dim=0)

        # Return sample
        return action_chunk, robot_obs, rgb_static, rgb_gripper, language_instruction

B = 1
calvin = CALVINDataset(dataset_dir)
calvin_dataloader = DataLoader(calvin, batch_size=B)

batch_action_chunk, batch_robot_obs, batch_rgb_static, batch_rgb_gripper, batch_instructions = next(iter(calvin_dataloader))
assert batch_action_chunk.shape == (B, 10, 7)
assert batch_robot_obs.shape == (B, 15)
assert batch_rgb_static.shape == (B, 3, 200, 200)
assert batch_rgb_gripper.shape == (B, 3, 84, 84)
assert len(batch_instructions) == B