"""play.py - Load the trained DQN model and watch it play Space Invaders."""

import argparse

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

DEFAULT_MODEL_PATH = "dqn_model.pth"
ENV_NAME = "ALE/SpaceInvaders-v5"
NUM_EPISODES = 3

gym.register_envs(ale_py)


class QNetwork(nn.Module):
    """CNN that maps 4 stacked frames (4x84x84) to one Q-value per action.
    IMPORTANT: this must be identical to the network in train.py."""

    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x / 255.0  # scale pixels to 0-1
        return self.fc(self.conv(x))


def make_env():
    """Same preprocessing as training: grayscale, 84x84, frame skip 4, stack 4."""
    env = gym.make(ENV_NAME, frameskip=1, render_mode="human")
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="path to the saved model")
    args = parser.parse_args()

    env = make_env()
    n_actions = env.action_space.n

    # Rebuild the network and load the trained weights
    model = QNetwork(n_actions)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()  # evaluation mode

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # Greedy action: always pick the highest Q-value (no epsilon)
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode}: reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
