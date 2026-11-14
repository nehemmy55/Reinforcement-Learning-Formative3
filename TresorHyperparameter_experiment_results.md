# Hyperparameter Experiment Results

**Member Name:** Tresor
**Game:** Space Invaders (`ALE/SpaceInvaders-v5`)
**Fixed settings for all 10 experiments:** 300,000 training steps, replay buffer 50,000, learning starts after 10,000 steps, train every 4 steps, target network update every 1,000 steps, seed 42. Trained on a laptop CPU (no GPU), so each run took between 67 and 158 minutes.

My baseline (experiment 1) is close to the original DQN paper: lr 0.0001, gamma 0.99, batch 32, epsilon from 1.0 to 0.05 over 100,000 steps. Every other experiment changes **one** value from the baseline, so any difference in score comes from that one change.

**How to read the numbers**
- **Best avg** = highest average game score over 20 consecutive training episodes.
- **Final avg** = average score over the last 20 training episodes.
- Scores are measured during training, so some actions are still random (epsilon-greedy).
- Each experiment was run once, so small differences (about 10 to 20 points) are within normal run-to-run noise.

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Baseline. Best avg 297.5, final avg 204.8. Slow, steady improvement (quarter averages 174 → 220), peaking late at episode 452. |
| 2 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Higher learning rate. Best avg 304.0, final avg 271.5. Learned faster early and peaked by episode 261, but scores swung more from episode to episode. |
| 3 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Lower learning rate. Best avg 278.0, final avg 204.3. The slowest start (first-quarter avg 162) and still climbing at the end, so 300,000 steps was too short for it. |
| 4 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 100,000 | Lower gamma. **Best run:** best avg 397.5 and final avg 391.8, both the highest of all 10. It improved in every quarter of training (166 → 285) and hit its peak in the very last episodes. |
| 5 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 100,000 | Higher gamma. Best avg 312.3, but final avg dropped to 196.0. It improved until the middle, then got worse in the last quarter (241 → 194). The far-future targets are noisy and made learning unstable. |
| 6 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 100,000 | Larger batch. Best avg 334.3 (second-best peak), reached early at episode 239, but final avg fell to 228.8. Also about twice as slow as the baseline (130 min). |
| 7 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 100,000 | Smaller batch. Best avg 274.0, the lowest of all runs, final avg 192.0. Noisy updates; scores flattened after the first half. |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Less random play at the end. Best avg 342.5, final avg 286.0. Highest single game (855). Dipped in the third quarter, then recovered. |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Faster epsilon decay. Best avg 303.5, final avg 226.5. About the same as the baseline; stopping exploration earlier neither helped nor hurt much. |
| 10 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 100,000 | Start with only 50% random actions. Best avg 281.3, final avg 202.0. Slightly below the baseline; less early exploration gave a weaker start to the replay buffer. |

## Best and worst configurations

- **Best: Experiment 4** (gamma 0.95). Best avg 397.5 and final avg 391.8, the highest on both measures, and still improving when training stopped. It is also the best result in the whole group.
- **Worst: Experiment 7** (batch size 16). Best avg 274.0 and final avg 192.0.
- Experiments 3 (lr 0.00005) and 10 (epsilon start 0.5) were also below the baseline.

## What helped and what harmed

- **Gamma had the biggest effect.** Lowering it to 0.95 gave by far the best run. In Space Invaders the points come from shooting aliens, which happens within a few steps, so a shorter horizon gives cleaner, less noisy targets. Raising gamma to 0.999 did the opposite: it peaked in the middle and then got worse.
- **Learning rate:** 0.0005 learned faster than the baseline and ended higher (final avg 271.5). 0.00005 was too slow for 300,000 steps.
- **Batch size:** 16 was too noisy and gave the worst result. 64 reached a good peak early but did not hold it, and doubled the training time on CPU.
- **Epsilon:** ending at 0.01 helped (best avg 342.5) because the agent played its learned policy more often. Decaying faster (50,000 steps) changed little, and starting at 0.5 hurt slightly.
- **Noise:** each configuration ran once with the same seed, so gaps of 10 to 20 points can be luck. The gap between experiment 4 and the others (about 60 points or more) is large enough to trust.

## Final configuration and justification

- **Learning rate 0.0001, gamma 0.95, batch size 32, epsilon 1.0 to 0.05 over 100,000 steps** (experiment 4).
- It had the highest best and final averages, and its curve was still going up at the end, so more training steps would likely help.
- This model is copied to `dqn_model.pth` at the repo root and is the one `play.py` loads by default.
- Limitation: one run per configuration. Repeating experiment 4 with other seeds would confirm the result, especially because Nehemie's low-gamma run (0.9) did not show the same gain.
