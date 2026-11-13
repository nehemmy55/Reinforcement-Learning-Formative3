# Hyperparameter Experiment Results

Member Name: Nehemie

Game: Space Invaders (`ALE/SpaceInvaders-v5`)

All 10 experiments used the same fixed settings: 300,000 training steps, replay buffer of 50,000, learning starts after 10,000 steps, training every 4 steps, target network update every 1,000 steps and seed 42. I trained them on a Kaggle GPU, each one took around 13 minutes.

My starting point (experiment 1) was lr 0.00025, gamma 0.99, batch 32 and epsilon going from 1.0 to 0.1 over 100,000 steps. In every other experiment I changed one or two values from this to see what happens.

Best avg is the highest average score over 20 episodes in a row. Final avg is the average of the last 20 episodes. These scores come from training, so the agent still takes some random actions.

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.00025 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Baseline. Best avg 348.5, final avg 191.8. It learned fast at the start but the score went down again near the end. |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | High learning rate. Best avg 308.0, final avg 192.8. It had the highest single game (975) but scores jumped up and down a lot and did not really improve over time. |
| 3 | 0.00001 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Very low learning rate. Best avg 294.5, final avg 212.0. Learning was very slow, it took almost twice as many episodes as the baseline to get going. |
| 4 | 0.00025 | 0.9 | 32 | 1.0 | 0.1 | 100,000 | Low gamma. Best avg 322.0, final avg 209.5. It had the shortest games, the agent only cared about points right now and did not avoid bombs well. |
| 5 | 0.00025 | 0.97 | 32 | 1.0 | 0.1 | 100,000 | Slightly lower gamma. Best avg 303.3, final avg 267.8. Lower peak but more stable than the baseline at the end. |
| 6 | 0.00025 | 0.99 | 128 | 1.0 | 0.1 | 100,000 | Big batch. Best avg 349.8, final avg 220.8. Good in the middle of training but dropped after. Also the slowest run (16.4 min). |
| 7 | 0.00025 | 0.99 | 8 | 1.0 | 0.1 | 100,000 | Very small batch. Best avg 262.3, final avg 218.3. Worst run. Updates were too noisy so it never learned much. |
| 8 | 0.00025 | 0.99 | 32 | 1.0 | 0.2 | 100,000 | Epsilon stops at 0.2. Best avg 361.5, final avg 326.5. Best run, it kept improving until the end even with 20% random moves. |
| 9 | 0.00025 | 0.99 | 32 | 1.0 | 0.1 | 200,000 | Slower epsilon decay. Best avg 355.3, final avg 290.8. Slow start because it was random for longer, but it kept getting better at the end. |
| 10 | 0.00025 | 0.99 | 64 | 1.0 | 0.02 | 150,000 | Mix: batch 64 with slower decay to a low epsilon. Best avg 361.3, final avg 324.5. Almost the same as experiment 8 and the score went up steadily. |

## Best and worst

- Best: experiment 8 (epsilon end 0.2). Best avg 361.5 and final avg 326.5, both the highest of my runs. Experiment 10 was almost the same (361.3 and 324.5).
- Worst: experiment 7 (batch size 8). Best avg only 262.3.
- Experiments 2 and 3 (learning rate too high or too low) were also weak.

## What helped and what hurt

- Exploration helped the most. My three best runs (8, 9 and 10) all explored more, either by keeping epsilon higher at the end or by decaying it more slowly. With epsilon ending at 0.1 after 100,000 steps (baseline) the agent seemed to stop trying new things too early and the score dropped at the end.
- Learning rate: 0.00025 worked best. 0.001 was too big and made learning unstable. 0.00001 was too small and learning was very slow.
- Gamma: 0.99 gave the best peak. With 0.9 the agent did not plan ahead, so it died faster (shortest episodes). 0.97 was in between.
- Batch size: 8 was too noisy and gave the worst result. 128 did not do better than 32 and was slower. 64 worked well in experiment 10.
- Each experiment was run only once, so differences of around 10 to 20 points can just be luck. The difference between the best runs (around 360) and the worst ones (260 to 300) is big enough to trust.

## Final configuration

I picked experiment 8: lr 0.00025, gamma 0.99, batch 32, epsilon from 1.0 to 0.2 over 100,000 steps.

It had the highest best and final average, and its score was still going up at the end of training, so more steps would probably help it. Experiment 10 is a close second. Note that play.py always picks the best action (no random moves), so the agent should do a bit better in the demo than these training scores show.
