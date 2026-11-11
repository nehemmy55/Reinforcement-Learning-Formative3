import argparse
import csv
import os
import random
import time
from collections import deque

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
AVERAGE_WINDOW = 20 # episodes used for the moving average reward


def make_env(render_mode=None):
    """Create Space Invaders with the Standard DQN preprocessing"""
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    """Maps a stack of 4 frames to one Q-value per action"""

    def __init__(self, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 9x9 -> 7x7
            nn.ReLU(),
            nn.Flatten(), # 64 * 7 * 7 = 3136 values
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0 # pixels 0..255 -> 0..1
        return self.head(self.features(x))


class ReplayBuffer:
    """Fixed size circular buffer of (obs, action, reward, next_obs, done)"""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.position = 0 # where the next transition will be written
        self.size = 0 # how many transitions are stored
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        i = self.position
        self.observations[i] = obs
        self.next_observations[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Return a random mini-batch as PyTorch tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return(
            torch.as_tensor(self.observations[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_observations[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )

def linear_epsilon(step, eps_start, eps_end, eps_decay_steps):
    progress = min(step / eps_decay_steps, 1.0)
    return eps_start + progress * (eps_end - eps_start)

def select_action(q_net, obs, epsilon, n_actions, device):
    """Random action with probability epsilon"""
    if random.random() < epsilon:
        return random.randrange(n_actions) # explore
    with torch.no_grad():
        obs_tensor = torch.as_tensor(np.array(obs), device=device).unsqueeze(0)
        return int(q_net(obs_tensor).argmax(dim=1).item()) # exploit


def optimize(q_net, target_net, optimizer, batch, gamma):
    obs, actions, rewards, next_obs, dones = batch

    # Q(s, a) predicted by the online network for the actions actually taken
    q_values = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: r + gamma * max_a ' Q_target(s', a')
    #(1 - done) removes the future part when the episode has ended
    with torch.no_grad():
        next_q_values = target_net(next_obs).max(dim=1).values
        targets = rewards + gamma * next_q_values * (1.0 - dones)

    #Huber loss is less sensitive to large errors than MSE (more stable)
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()

def save_episode_log(episode, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode[0].keys()))
        writer.writeheader()
        writer.writerows(episode)

def moving_average(values, window):
    return [np.mean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]

def save_training_plot(episodes, path, title):
    rewards = [e["reward"] for e in episodes]
    lengths = [e["episode_length"] for e in episodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(rewards, alpha=0.3, label="Episode reward")
    ax1.plot(moving_average(rewards, AVERAGE_WINDOW), label=f"{AVERAGE_WINDOW}-episode average")
    ax1.set_ylabel("Reward")
    ax1.set_title(title)
    ax1.legend()

    ax2.plot(lengths, alpha=0.3, label="Episode length")
    ax2.plot(moving_average(lengths, AVERAGE_WINDOW), label=f"{AVERAGE_WINDOW}-episode average")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def append_summary(args, episodes, best_avg_reward, minutes, path):
    """Add one row per experiment to summary"""
    final_rewards = [e["reward"] for e in episodes[-AVERAGE_WINDOW:]]
    row = {
        "exp": args.exp,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_decay": args.eps_decay,
        "total_steps": args.total_steps,
        "episodes": len(episodes),
        "best_avg_reward": round(best_avg_reward, 2),
        "final_avg_reward": round(float(np.mean(final_rewards)), 2),
        "mean_episode_length": round(float(np.mean([e["episode_length"] for e in episodes])), 1),
        "train_minutes": round(minutes, 1),
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)



def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape # (4, 84, 84)

    # Online network (trained every step) and target network (a frozen copy)
    q_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, obs_shape)

    out_dir = os.path.join("models", args.member)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"exp{args.exp}_dqn_model.pth")

    print(f"Experiment {args.exp} | device={device} | {vars(args)}")
    start_time = time.time()

    episodes = []
    recent_rewards = deque(maxlen=AVERAGE_WINDOW)
    best_avg_reward = -float("inf")
    episode_reward, episode_length = 0.0, 0
    obs, _ = env.reset(seed=args.seed)

    for step in range(1, args.total_steps + 1):
        #Act in the environment and remember what happened
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(q_net, obs, epsilon, n_actions, device)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Rewards are clipped to -1/0/+1 for training
        # but the real game score is what we log
        buffer.add(obs, action, np.sign(reward), next_obs, terminated)
        obs = next_obs
        episode_reward += reward
        episode_length += 1

        #Learn from a random mini-batch every `train_freq` steps.
        if step > args.learning_starts and step % args.train_freq == 0:
            batch = buffer.sample(args.batch_size, device)
            optimize(q_net, target_net, optimizer, batch, args.gamma)

        # Copy the online weights into the target network periodically
        if step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if terminated or truncated:
            episodes.append({
                "episode": len(episodes) + 1,
                "total_steps": step,
                "episode_length": episode_length,
                "reward": episode_reward,
                "epsilon": round(epsilon, 4),
            })
            recent_rewards.append(episode_reward)
            avg_reward = float(np.mean(recent_rewards))

            # keep the weights with the best moving-average reward.
            if len(recent_rewards) == AVERAGE_WINDOW and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(q_net.state_dict(), model_path)

            if len(episodes) % 10 == 0:
                print(
                    f"step {step:>8} | episode {len(episodes):>4} | "
                    f"reward {episode_reward:6.1f} | avg{AVERAGE_WINDOW} {avg_reward:6.1f} | "
                    f"eps {epsilon:.3f}"
                )

            obs, _ = env.reset()
            episode_reward, episode_length = 0.0, 0
    env.close()

        # Very short runs may never complete AVERAGE_WINDOW episodes
    if best_avg_reward == -float("inf"):
            torch.save(q_net.state_dict(), model_path)
            best_avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0

    minutes = (time.time() - start_time) / 60
    if episodes:
            save_episode_log(episodes, os.path.join(out_dir, f"exp{args.exp}_log.csv"))
            save_training_plot(
                episodes,
                os.path.join(out_dir, f"exp{args.exp}_plot.png"),
                title=f"{args.member} - experiment {args.exp}",
            )
            append_summary(args, episodes, best_avg_reward, minutes, os.path.join (out_dir, "summary.csv"))

    print(f"Done in {minutes:.1f} min. Best {AVERAGE_WINDOW}-episode average reward: {best_avg_reward:.1f}")
    print(f"Model saved to {model_path}")

def parse_args(argv=None):
        p = argparse.ArgumentParser(description="Train a DQN on Space Invaders")
        p.add_argument("--member", required=True, help="your name; results go to models/<member>/")
        p.add_argument("--exp", type=int, required=True, help="experiment number (1-10)")

        # Hyperparameters investigated in Task 3
        p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--eps-start", type=float, default=1.0)
        p.add_argument("--eps-end", type=float, default=0.05)
        p.add_argument("--eps-decay", type=int, default=100_000, help="steps to go from eps-start to eps-end")

        #Fixed settings (kept the same for every experiment).
        p.add_argument("--total-steps", type=int, default=300_000)
        p.add_argument("--buffer-size", type=int, default=50_000)
        p.add_argument("--learning-starts", type=int, default=10_000, help="random steps before learning")
        p.add_argument("--train-freq", type=int, default=4, help="learn every N environment steps")
        p.add_argument("--target-update", type=int, default=1_000, help="copy weights every N steps")
        p.add_argument("--seed", type=int, default=42)
        return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
