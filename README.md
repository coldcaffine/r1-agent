# Reinforcement learning from scratch

Four small projects that build up from a lookup table to a neural-network agent that plays Snake.

## Files

| File | What it does | Result |
|---|---|---|
| `agent.py` | Q-learning on Gymnasium's FrozenLake (no slipping) | 100/100 goal reaches |
| `slippery.py` | Same agent on slippery ice, tuned (slower exploring, small learning rate) | 758/1000, close to the ~74-75% a perfect policy can get |
| `gridworld.py` | My own 6x6 grid with traps, Q-table | shortest safe path found |
| `gridworld_dqn.py` | Same grid, Q-table replaced by a neural network (DQN) | 30/30 starting tiles reach the goal |
| `snake.py` | DQN playing Snake on an 8x8 board | average 30.0 food over 100 games, best 49 |

## What I learned from the experiments

- With zero exploring the agent never finds the goal and learns nothing.
- With gamma = 0 there is no future reward, so only the tile next to the goal gets a value.
- A learning rate of 1.0 works on deterministic ice but wipes out every value on slippery ice.
- The DQN reached the goal from only 26/30 tiles when it always started in one corner, because it had barely seen the other tiles. Starting episodes on random tiles exposed a second bug: I was treating a 50-step time cutoff as a real ending, so the network learned that looping into a wall was cheap. Only reaching the goal or a trap should count as an ending.
- Snake with 11 simple inputs averaged 16.0 food, then died by walking into dead ends it could not see. Adding 3 inputs (reachable free space after each possible move, by flood fill) raised the average to 30.0. Training twice as long gave 30.9, no real gain.

Numbers come from single training runs, so they vary between runs.

## Run it

    pip install -r requirements.txt
    python agent.py
    python gridworld.py
    python gridworld_dqn.py
    python snake.py

`snake.py` trains for a few minutes, tests 100 games, then replays one game in the terminal.
