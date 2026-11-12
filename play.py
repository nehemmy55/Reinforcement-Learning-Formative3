"""play.py - Load the trained DQN model and watch it play Space Invaders."""

import argparse

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
DEFAULT_MODEL_PATH = "dqn_model.pth"
NUM_EPISODES = 3


def make_env(render_mode=None):
    """Same preprocessing as train.py: grayscale, 84x84, frame skip 4, stack 4."""
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    """Maps a stack of 4 frames to one Q-value per action.
    This is identical to the QNetwork in train.py (same layers, same names)."""

    def __init__(self, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9x9 -> 7x7
            nn.ReLU(),
            nn.Flatten(),  # 64 * 7 * 7 = 3136 values
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0  # pixels 0..255 -> 0..1
        return self.head(self.features(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="path to the saved model")
    args = parser.parse_args()

    env = make_env(render_mode="human")
    n_actions = env.action_space.n

    # Rebuild the network and load the trained weights
    model = QNetwork(n_actions)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()  # evaluation mode

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # Greedy action: always pick the highest Q-value (no epsilon)
            obs_tensor = torch.as_tensor(np.array(obs)).unsqueeze(0)
            with torch.no_grad():
                action = int(model(obs_tensor).argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode}: reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
