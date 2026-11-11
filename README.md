# DQN on Atari Space Invaders

A Deep Q-Network (DQN) agent implemented from scratch with PyTorch and Gymnasium, trained to play Atari Space Invaders. No ready-made RL libraries (Stable-Baselines3, RLlib, etc.) are used.

## Group Members

| Name | Role |
|---|---|
|nehemie| |
| tresor| |
| sonia| |


## Selected Environment: Space Invaders

**Environment ID:** `ALE/SpaceInvaders-v5` (Gymnasium + Arcade Learning Environment)

**Objective of the game:** The player controls a laser cannon at the bottom of the screen. Rows of aliens move side to side and slowly descend while dropping bombs. The goal is to shoot as many aliens as possible to earn points, while dodging bombs and hiding behind the shields. The game ends when all lives are lost or the aliens reach the bottom.

**Actions (6):**

| ID | Action |
|---|---|
| 0 | NOOP (do nothing) |
| 1 | FIRE |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | RIGHT + FIRE |
| 5 | LEFT + FIRE |

**Reward:** points for each alien destroyed (aliens in higher rows are worth more; the mystery ship gives bonus points).

### Preprocessing

The raw game frames (210×160 RGB) are prepared with the standard DQN preprocessing (`make_env()` in `train.py`):

| Step | Why |
|---|---|
| Convert to grayscale | Colour is not needed to make decisions; 3× less data |
| Resize to 84×84 | Smaller input → faster network |
| Frame skip = 4 | The chosen action is repeated for 4 frames, speeding up training |
| Stack the last 4 frames | A single frame cannot show movement; 4 frames let the agent see the direction of aliens and bullets |

Final observation given to the network: **4 × 84 × 84** (uint8 pixels).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python and package management.

```bash
uv sync
```

## Training

```bash
uv run train.py --member <YourName> --exp 1 --lr 1e-4 --gamma 0.99 --batch-size 32 \
    --eps-start 1.0 --eps-end 0.05 --eps-decay 100000 --total-steps 300000
```

Outputs are saved to `models/<YourName>/`:
- `exp<N>_dqn_model.pth` – model with the best 20-episode average reward
- `exp<N>_log.csv` – episode, total steps, episode length, reward, epsilon
- `exp<N>_plot.png` – reward and episode-length curves

## Playing

```bash
uv run play.py                                         # loads dqn_model.pth
uv run play.py --model models/<YourName>/exp1_dqn_model.pth
```

## DQN Architecture

_To be added: network diagram and explanation of the training process._

## Hyperparameter Experiments

_To be added: link to each member's results table._

## Gameplay Demonstration

_To be added: video of the best trained agent._
