# Hyperparameter Experiment Results

**Member Name:** Sonia
**Game:** Space Invaders (`ALE/SpaceInvaders-v5`)
**Fixed settings for all 10 experiments:** 300,000 training steps, replay buffer 50,000, learning starts after 10,000 steps, train every 4 steps, target network update every 1,000 steps, seed 42.

**How to read the numbers**
- **Best avg** = highest average game score over 20 consecutive training episodes.
- **Final avg** = average score over the last 20 training episodes.
- Scores are measured during training, so some actions are still random (epsilon-greedy).
- Each experiment was run once, so small differences (about 10 to 20 points) are within normal run-to-run noise.

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0002 | 0.99 | 32 | 1.0 | 0.03 | 75,000 | Baseline. Best avg 316.5, final avg 270.3. Improved steadily as epsilon fell. |
| 2 | 0.00075 | 0.99 | 32 | 1.0 | 0.03 | 75,000 | Higher learning rate. Best avg 313.5 (similar to baseline) but final avg fell to 223.3, so it ended less stable. |
| 3 | 0.000025 | 0.99 | 32 | 1.0 | 0.03 | 75,000 | Much lower learning rate. Best avg 325.8, final avg 254.8. Still learned well in 300,000 steps. |
| 4 | 0.0002 | 0.98 | 32 | 1.0 | 0.03 | 75,000 | Lower gamma. Best avg 284.5, the lowest best average of all runs. Final avg 228.5. |
| 5 | 0.0002 | 0.995 | 32 | 1.0 | 0.03 | 75,000 | Higher gamma. Best avg 296.0, final avg 235.8. Slightly below the baseline. |
| 6 | 0.0002 | 0.99 | 96 | 1.0 | 0.03 | 75,000 | Larger batch. **Best configuration:** best avg 370.0 and the highest final avg (311.8). Took slightly longer (16.3 min). |
| 7 | 0.0002 | 0.99 | 24 | 1.0 | 0.03 | 75,000 | Smaller batch. Best avg 285.8, final avg 242.5. Scores dipped for a stretch late in training. |
| 8 | 0.0002 | 0.99 | 32 | 1.0 | 0.15 | 75,000 | Epsilon kept at 0.15. Best avg 323.5, but the lowest final avg (191.5) because 15% of actions stayed random. |
| 9 | 0.0002 | 0.99 | 32 | 1.0 | 0.03 | 40,000 | Faster epsilon decay. Best avg 322.3, final avg 287.0. About the same as the baseline. |
| 10 | 0.0002 | 0.99 | 32 | 0.8 | 0.03 | 75,000 | Lower starting epsilon. Best avg 324.8, final avg 291.0. About the same as the baseline. |

## Best and worst configurations

- **Best: Experiment 6** (batch size 96). It had the highest best average (370.0) and the highest final average (311.8).
- **Worst by best average: Experiment 4** (gamma 0.98, best avg 284.5). Experiment 7 (batch size 24, 285.8) was almost the same.
- **Worst by final average: Experiment 8** (epsilon end 0.15, final avg 191.5).

## What helped and what harmed

- **Batch size had the clearest effect.** A larger batch (96) helped, and a smaller batch (24) hurt. A larger batch averages over more experiences per update, which gives a less noisy learning signal.
- **Gamma:** both 0.98 and 0.995 scored below the baseline of 0.99, so 0.99 looks like a good middle value.
- **Learning rate:** 0.000025 and 0.0002 reached similar best averages. The higher rate (0.00075) ended less stable.
- **Epsilon end:** keeping epsilon high (0.15) did not lower the best average but hurt the final average, since the agent kept acting randomly.
- **Epsilon start and decay:** starting at 0.8 or decaying faster (40,000 steps) gave about the same results as the baseline.
- **Noise check:** experiment 1 was run twice by accident. One run gave a best average of 324.5 and the saved run gave 316.5. This shows that differences of about 10 points are noise, so only the batch size results (and possibly gamma) are strong enough to draw conclusions from.

## Final configuration and justification

- **Learning rate 0.0002, gamma 0.99, batch size 96, epsilon 1.0 to 0.03 over 75,000 steps** (experiment 6).
- It had the best score on both measures, and the larger batch is consistent with a more stable update.
- Limitation: this is one run per configuration. A repeat with other seeds would be needed to confirm the gap.
