# Reinforcement Learning From Scratch

A self-learning agent in Python, built in stages: a lookup-table agent, my own grid world, a neural-network agent (DQN), and finally an agent that learns to play Snake.

Nobody tells the agent the rules or shows it a solution. It tries things, gets rewards, and improves on its own. This README explains every concept, every file, every experiment I ran, the results, and the bugs I found along the way.

> All numbers come from single training runs. Training is random, so your results will differ a little.

---

## Table of contents

1. [What is reinforcement learning?](#1-what-is-reinforcement-learning)
2. [Project files](#2-project-files)
3. [Part 1: Q-learning on FrozenLake (`agent.py`)](#3-part-1-q-learning-on-frozenlake-agentpy)
4. [Part 2: Experiments on the hyperparameters](#4-part-2-experiments-on-the-hyperparameters)
5. [Part 3: Slippery ice (`slippery.py`)](#5-part-3-slippery-ice-slipperypy)
6. [Part 4: My own environment (`gridworld.py`)](#6-part-4-my-own-environment-gridworldpy)
7. [Part 5: Replacing the table with a neural network (`gridworld_dqn.py`)](#7-part-5-replacing-the-table-with-a-neural-network-gridworld_dqnpy)
8. [Part 6: Snake (`snake.py`)](#8-part-6-snake-snakepy)
9. [Bugs I found and fixed](#9-bugs-i-found-and-fixed)
10. [How to run everything](#10-how-to-run-everything)
11. [How to reproduce the experiments](#11-how-to-reproduce-the-experiments)
12. [What I learned](#12-what-i-learned)

---

## 1. What is reinforcement learning?

Reinforcement learning (RL) is learning by trial and error. There are five ideas:

| Term | Meaning | Example in this project |
|---|---|---|
| **Environment** | The world the agent lives in | FrozenLake, my grid world, Snake |
| **State** | What the agent can see right now | Which tile it is on |
| **Action** | What the agent can do | Move left, down, right, up |
| **Reward** | Feedback after each action | +1 for reaching the goal, -10 for falling in a trap |
| **Episode** | One full attempt from start to finish | One game |

The agent's job is to find a strategy (called a **policy**) that collects as much reward as possible over time.

---

## 2. Project files

| File | What it does | Result |
|---|---|---|
| `agent.py` | Q-learning on FrozenLake (non-slippery), using a Q-table | 100/100 goal reaches |
| `slippery.py` | Same agent on slippery ice, with tuned settings | 758/1000 goal reaches |
| `gridworld.py` | My own 6x6 grid world with traps, solved with a Q-table | Shortest safe path found (reward 1) |
| `gridworld_dqn.py` | Same grid world, but the Q-table is replaced by a neural network (DQN) | Goal reached from 30/30 starting tiles |
| `snake.py` | A DQN agent that plays Snake on an 8x8 board | Average 30.0 food over 100 games, best 49 |
| `snake.pt` | Saved weights of the trained Snake network | - |
| `requirements.txt` | Libraries needed: `gymnasium`, `numpy`, `torch` | - |

---

## 3. Part 1: Q-learning on FrozenLake (`agent.py`)

### The environment

FrozenLake is a 4x4 grid. `S` = start, `F` = safe ice, `H` = hole, `G` = goal.

```
S F F F      tile numbers:   0  1  2  3
F H F H                      4  5  6  7
F F F H                      8  9 10 11
H F F G                     12 13 14 15
```

The agent only sees its tile number (0-15). It has 4 actions: 0 = left, 1 = down, 2 = right, 3 = up. Reaching the goal gives +1. Everything else gives 0. Falling in a hole ends the episode.

### The Q-table: the agent's brain

`Q` is a table with one row per tile and one column per action. Each cell answers: *"If I'm on this tile and take this action, how good is my future?"*

It starts as all zeros, so the agent knows nothing. Training fills in the numbers. After training, the agent looks at its row and picks the column with the highest number.

### The learning formula

After every action, the table is updated with this one line:

```
Q[s, a] = Q[s, a] + alpha * (reward + gamma * max(Q[next_s]) - Q[s, a])
```

Read it as: **new estimate = old estimate + a small step toward a better estimate.**

- `reward + gamma * max(Q[next_s])` is the **target**: the reward I just got, plus the best future I expect from the next tile, discounted.
- `target - Q[s, a]` is the **error**: how wrong my old estimate was.
- `alpha` is how big a step I take toward fixing the error.

**Worked example** (`alpha = 0.1`, `gamma = 0.99`, all values start at 0):

1. The agent is on tile 14 and moves right into the goal. Reward = 1. The goal tile is terminal, so `max(Q[15]) = 0`.
   `Q[14, right] = 0 + 0.1 * (1 + 0 - 0) = 0.1`
2. Later, the agent is on tile 10 and moves down to tile 14. Reward = 0, but `max(Q[14])` is now 0.1.
   `Q[10, down] = 0 + 0.1 * (0 + 0.99 * 0.1 - 0) = 0.0099`

The reward at the goal leaks backward, one step at a time, to every tile that leads toward it. After thousands of episodes the whole table has a trail of rising values leading to the goal.

### The settings (hyperparameters)

| Setting | Value | Meaning |
|---|---|---|
| `alpha` | 0.1 | Learning rate: size of each update step |
| `gamma` | 0.99 | Discount: how much future rewards matter (0 = only now, 1 = long-term) |
| `epsilon` | starts 1.0 | Exploration rate: chance of a random move |
| `epsilon_min` | 0.05 | Never go below 5% random |
| `epsilon_decay` | 0.999 | `epsilon` is multiplied by this after every episode |
| `episodes` | 5000 | Number of training games |

### Explore vs exploit

If the agent always chose the best-looking action, it would never discover better ones. So it uses **epsilon-greedy**: with probability `epsilon` it picks a random action (explore), otherwise the best known action (exploit). It starts at 100% random and shrinks `epsilon` each episode, so it explores a lot early and trusts what it learned later.

### Result

```
Episode 500:  wins in last 500 = 52
Episode 2000: wins in last 500 = 403
Episode 5000: wins in last 500 = 476
Test: reached the goal 100/100 times
```

The learned policy (arrows show the best action for each tile):

```
→ → ↓ ←
↑ ← ↓ ←
→ → ↓ ←
← → → ←
```

Following it from the top-left: right, right, down, down, down, right into the goal, the shortest safe path (6 moves). Arrows on holes and the goal mean nothing, because the agent never acts from those tiles. In the Q-table, row 14 (the tile before the goal) has `Q[14, right] = 1.0`, and the values shrink by about 1% per step going backward: 0.99, 0.98, 0.97... That is the trail created by `gamma = 0.99`.

---

## 4. Part 2: Experiments on the hyperparameters

I changed one setting at a time on `agent.py` to see what each one does. I made a copy of the file and edited one line each time.

| # | Change | Result | What it showed |
|---|---|---|---|
| 1 | `epsilon = 0` (no exploring) | 0 wins, Q-table all zeros, **0/100** | The agent never found the goal even once, so there was nothing to learn from. All values tied at 0, so it always picked "left". Exploration is required. |
| 2 | `gamma = 0.5` | **100/100** | Values halve at every step away from the goal (1.0 → 0.5 → 0.25...), reaching about 0.03 at the start. Tiny, but `argmax` only cares about the order. Still works on a small grid. |
| 3 | `gamma = 0.0` | **0/100** | Only one non-zero value in the whole table: `Q[14, right] = 0.469`. With no future reward, nothing leaks backward, so the agent has no trail. That 0.469 is just a count of the 6 lucky wins (1 − 0.9⁶). |
| 4a | `alpha = 1.0` | **100/100** | Each update fully replaces the old value. On a world with no randomness, the new value is always correct, so this works. |
| 4b | `alpha = 0.001` | **100/100** | Tiny steps, but the tile before the goal was updated about 3,900 times, so values near the goal got large. Far tiles stayed near 0, but the agent never needed them. |

**Lesson:** In an easy, non-random world, `gamma` and `alpha` barely matter. They matter when the world is random (next part). Also, my early predictions for experiments 2 and 4b were wrong: I expected `gamma = 0.5` and `alpha = 0.001` to fail, and they did not. Running the experiment beats guessing.

---

## 5. Part 3: Slippery ice (`slippery.py`)

With `is_slippery=True`, a move only goes the intended way about 1/3 of the time and slides sideways otherwise. The world is now random, even a perfect policy only wins about 74-75% of the time.

| Run | Settings | Result |
|---|---|---|
| 5a | `alpha = 0.1`, 20,000 episodes | **49/100** |
| 5b | `alpha = 1.0`, 20,000 episodes | **0/100** (total collapse) |
| 6 | `alpha = 0.05`, `epsilon_decay = 0.9998`, 30,000 episodes, tested on 1,000 runs | **758/1000** (75.8%) |

**Why 5b collapsed:** with `alpha = 1.0` every update completely overwrites the old value. On slippery ice, a good move sometimes slides into a hole, and that one unlucky result erased the good value. After enough of those, the whole Q-table was zeros, and the agent could never recover.

**Why 5a was stuck at 49%:** exploration ended too early (epsilon hit its floor around episode 3000), and a constant `alpha = 0.1` kept the values wobbling because each outcome is random.

**The fix (run 6):** explore for much longer (slower decay), use a smaller learning rate so one unlucky outcome cannot swing a value, and train longer. The score of 75.8% is essentially the best possible on this environment.

The learned arrows look odd. For example, tile 0 points left into the wall. That is correct on ice: pushing into a wall is the safest move because any sideways slide from there stays on safe tiles. The agent discovered this on its own.

`slippery.py` is `agent.py` with `is_slippery=True`, 30,000 episodes, `alpha = 0.05`, `epsilon_decay = 0.9998`, and a 1,000-game test.

---

## 6. Part 4: My own environment (`gridworld.py`)

Until now the world came from a library. Here I wrote the world myself as a Python class.

- **Board:** 6x6 grid. Start top-left (0,0), goal bottom-right (5,5).
- **Traps:** 5 traps at (1,1), (2,3), (3,1), (4,4), (1,4).
- **Actions:** left, down, right, up. Walking into a wall keeps you on the same tile.
- **Rewards:** -1 for each step (this makes the agent want short paths), +10 for the goal, -10 for a trap.
- **Time limit:** 50 steps per episode.
- **State:** the tile number, 0-35 (`row * 6 + column`).

The class has two methods the agent uses: `reset()` (start a new episode) and `step(action)` (returns the new state, the reward, and whether the episode ended).

### Result

```
→ ↓ ... (arrows trace a route to G avoiding every X)
Path taken: (0,0) → (1,0) → (2,0) → (2,1) → (2,2) → (3,2) → (4,2) → (4,3) → (5,3) → (5,4) → (5,5)
Total reward: 1 (best possible is 1)
```

The best possible reward is 1: 10 moves, nine at -1, then +10 for the goal.

### Follow-up tests

- **Blocked the old route** by adding traps at (2,2) and (4,2). The agent found a completely new 10-move path along the left edge and bottom row, still reward 1.
- **Changed the rewards** to goal = +1 and step cost = 0. I expected the path to wander, but it stayed a shortest 10-move path. The reason: `gamma = 0.99` quietly acts as a step penalty, because a goal 10 steps away is worth more (about 0.91) than one 14 steps away (about 0.87).

---

## 7. Part 5: Replacing the table with a neural network (`gridworld_dqn.py`)

### Why

A Q-table needs one row for every possible state. That works for 36 tiles, but not for a game screen or a robot, which have billions of states. A **Deep Q-Network (DQN)** replaces the table with a neural network that *estimates* the Q-values from the state, and it generalizes to states it has not seen exactly.

### What is different from the Q-table version

| Idea | What it does |
|---|---|
| **The network** | Takes the position `(row/5, col/5)` as input and outputs 4 Q-values. Two hidden layers of 64 neurons with ReLU. |
| **Replay buffer** | Stores the last 10,000 steps. Each update trains on a random batch of 64 old steps, which stops the network from overfitting to the last few moves. |
| **Target network** | A slowly updated copy of the network (refreshed every 200 steps) used to compute the target `reward + gamma * max(Q_next)`, so the target does not shift on every step. |
| **Loss + optimizer** | Instead of `Q += alpha * error`, the network minimizes `(Q_predicted − target)²` using the Adam optimizer. It is the same idea, but the learning is done by gradient descent. |

### The results, step by step

| Version | Change | Test from every safe tile |
|---|---|---|
| `dqn.py` | Every episode starts at (0,0) | **26/30** |
| `dqn2.py` | Episodes start on a random safe tile | **25/30** (worse, see below) |
| `dqn3.py` = `gridworld_dqn.py` | Random starts + fixed ending bug, 5,000 episodes | **30/30** |

The first version solved the path from (0,0), but when I tested it from all 30 safe tiles it failed on 4. Its arrows on tiles it rarely visited pointed into traps. It had barely trained on those tiles, so it was guessing. A table would show 0 there. A network guesses confidently, which is harder to notice. **A trained agent should be tested from many situations, not just the one it practiced.**

Training from random start tiles (called exploring starts) should have helped, but it made things worse and exposed a bug. See the next section.

---

## 8. Part 6: Snake (`snake.py`)

### Why a table cannot work

The number of possible snake shapes on a board is enormous, so every situation cannot have its own row. The network must generalize.

### The game

- **Board:** 8x8. The snake starts with length 1 in the middle.
- **Actions (3):** go straight, turn right, turn left.
- **Rewards:** +10 for eating food, -10 for dying (hitting a wall or itself), 0 otherwise.
- **Time limit:** if 100 steps pass without eating, the game is stopped (not counted as a death).
- **Winning:** filling the whole board (63 food).

### What the agent sees

Instead of the whole board, the agent gets a short list of yes/no numbers:

**Version 1: 11 inputs**
- 3 danger flags: is there a wall/body straight ahead, to the right, to the left?
- 4 flags for the current direction (up, right, down, left)
- 4 flags for where the food is (up, right, down, left)

**Version 2: 14 inputs** adds 3 numbers: for each of the 3 possible moves, how many free cells the head could still reach after making that move (counted with a flood fill, scaled 0-1).

### The network and settings

- Network: 14 inputs → 128 → 128 → 3 outputs (ReLU).
- `gamma = 0.95`, `lr = 1e-3`, replay buffer of 50,000 steps, batch of 64, target network refreshed every 500 steps.
- `epsilon` starts at 1.0, decays by 0.995 per episode to a floor of 0.01.
- 1,500 training episodes.

### Results

| Version | Training curve | Test over 100 games |
|---|---|---|
| 11 inputs | 0.47 average food at episode 100, rising to about 16 and then flat | average **16.0**, best **28** |
| 14 inputs (flood fill) | – | average **30.0**, best **49** |
| 14 inputs, 3,000 episodes, slower exploring | still rising slowly and noisy | average **30.9**, best 48 (no real gain) |

A random snake eats about 0.3 food, so even the first version learned a lot.

### Finding the problem by watching it play

I replayed one game of the 11-input snake. It died in a spiral: it walked into a narrow gap between its own body, and the gap was closed at the other end. With only one-cell lookahead, every step into the gap looked safe, until the head was inside and every neighbor was body. That is why the training curve flattened at 15-17: the agent could not see dead ends.

Adding the flood-fill inputs (version 2) let the snake see how much open space each move leads to. A move into a tiny pocket scores near 0, and a move into open space scores high. That nearly doubled the average score (16.0 → 30.0).

Training twice as long did not help (30.9). At 3,000 episodes the learning curve was still rising slowly, and by that point extra training mostly added noise. Getting past about 30 would need something different, like better inputs or another algorithm. The final model is the 1,500-episode, 14-input version.

---

## 9. Bugs I found and fixed

### Bug 1: Treating a time cutoff as a real ending (DQN grid world)

When I started episodes from random tiles, the DQN got worse (25/30), and the agent stood at (0,0) bumping into the wall for all 50 steps.

**Cause:** in the learning target `reward + gamma * max(Q_next) * (1 − done)`, I set `done = True` when the 50-step limit ended the episode. That told the network "nothing comes after this," so an endless wall-bumping loop only cost -1 in total, and the network learned that wall-bumping is cheap. Starting always at (0,0) had hidden this.

**Fix:** only reaching the goal or a trap is a real ending. A time cutoff must still look ahead. In RL terms this is the difference between **terminated** (the game really ended) and **truncated** (we just stopped it). After the fix the result went from 25/30 to **30/30**. The Snake code handles this correctly from the start (`terminated` and `truncated` are separate).

### Bug 2: The Snake's blind spot

Described above: the 11-input snake could not see dead ends. Fixed with flood-fill inputs.

### Bug 3: A DQN is confidently wrong on unseen states

Described in Part 5. Fixed with random starts, and by always testing from many starting situations.

---

## 10. How to run everything

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install the libraries
pip install -r requirements.txt

# 3. Run the projects
python agent.py             # Q-learning on FrozenLake
python slippery.py          # slippery ice (takes a minute)
python gridworld.py         # my grid world with a Q-table
python gridworld_dqn.py     # grid world with a neural network
python snake.py             # trains Snake, tests 100 games, replays one game
```

Notes:
- `snake.py` trains for a few minutes, then prints `Test over 100 games: average food = ..., best = ...`, then replays one game in the terminal (`O` = head, `o` = body, `F` = food). Running it retrains the network and overwrites `snake.pt`.
- Python 3.14 was used. If `torch` fails to install on your Python version, use an older Python such as 3.12.

---

## 11. How to reproduce the experiments

On macOS, each experiment is a copy of `agent.py` with one line changed (on Linux, remove the `''` after `-i`):

```bash
# Experiment 1: no exploring
cp agent.py exp1.py
sed -i '' 's/^epsilon = 1.0/epsilon = 0.0/; s/^epsilon_min = 0.05/epsilon_min = 0.0/' exp1.py
python exp1.py

# Experiment 2: gamma = 0.5
cp agent.py exp2.py
sed -i '' 's/^gamma = 0.99/gamma = 0.5/' exp2.py
python exp2.py

# Experiment 3: gamma = 0
cp agent.py exp3.py
sed -i '' 's/^gamma = 0.99/gamma = 0.0/' exp3.py
python exp3.py

# Experiment 4a / 4b: alpha = 1.0 and alpha = 0.001
cp agent.py exp4a.py
sed -i '' 's/^alpha = 0.1/alpha = 1.0/' exp4a.py
python exp4a.py

cp agent.py exp4b.py
sed -i '' 's/^alpha = 0.1/alpha = 0.001/' exp4b.py
python exp4b.py

# Experiment 5a: slippery ice
cp agent.py exp5a.py
sed -i '' 's/is_slippery=False/is_slippery=True/; s/^episodes = 5000/episodes = 20000/' exp5a.py
python exp5a.py
```

---

## 12. What I learned

- **Exploration is essential.** Without it the agent never finds the reward and learns nothing.
- **`gamma` creates a trail of values** from the goal back to the start, and it also works as a hidden step penalty.
- **The learning rate must match the world.** `alpha = 1.0` is fine without randomness and destroys everything with randomness.
- **Tune for the random case:** slower exploration decay + a smaller `alpha` + more episodes took slippery ice from 49% to about the best possible.
- **Test from many situations.** The DQN looked perfect from one start tile and failed on 4 of 30 tiles.
- **Terminated vs truncated** matters. Counting a time limit as a real ending teaches the agent the wrong thing.
- **What the agent can see limits how well it can play.** Three extra inputs (reachable space) nearly doubled the Snake's score.
- **Watching the agent play** showed me a problem (the dead-end spiral) that the numbers alone only hinted at.
- **More training is not always better.** Doubling the Snake's training gave no real improvement.

## Tools

Python, NumPy, Gymnasium (FrozenLake), PyTorch (DQN).