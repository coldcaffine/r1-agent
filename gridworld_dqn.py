import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class GridWorld:
    """Same 6x6 world, but the state is two numbers: (row, col) scaled to 0-1."""

    def __init__(self):
        self.size = 6
        self.goal = (5, 5)
        self.traps = {(1, 1), (2, 3), (3, 1), (4, 4), (1, 4)}
        self.max_steps = 50

    def reset(self, random_start=False):
        self.pos = (0, 0)
        if random_start:
            while True:
                self.pos = (random.randrange(6), random.randrange(6))
                if self.pos not in self.traps and self.pos != self.goal:
                    break
        self.steps = 0
        return self._obs()

    def _obs(self):
        return np.array([self.pos[0] / 5, self.pos[1] / 5], dtype=np.float32)

    def step(self, action):
        r, c = self.pos
        if action == 0:
            c -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c += 1
        elif action == 3:
            r -= 1
        r = min(max(r, 0), self.size - 1)
        c = min(max(c, 0), self.size - 1)
        self.pos = (r, c)
        self.steps += 1

        if self.pos == self.goal:
            return self._obs(), 10, True
        if self.pos in self.traps:
            return self._obs(), -10, True
        if self.steps >= self.max_steps:
            return self._obs(), -1, True
        return self._obs(), -1, False


# ---------- THE NETWORK ----------
net = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 4),
)
target_net = copy.deepcopy(net)  # slow-moving copy used for targets
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# ---------- SETTINGS ----------
env = GridWorld()
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.998
episodes = 5000
batch_size = 64
target_update_every = 200  # steps
buffer = deque(maxlen=10000)
total_steps = 0

# ---------- TRAINING ----------
for episode in range(episodes):
    obs = env.reset(random_start=True)
    done = False
    episode_reward = 0

    while not done:
        # choose action (epsilon-greedy, same as before)
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            with torch.no_grad():
                action = net(torch.tensor(obs)).argmax().item()

        next_obs, reward, done = env.step(action)
        buffer.append((obs, action, reward, next_obs, env.pos == env.goal or env.pos in env.traps))  # remember it
        obs = next_obs
        episode_reward += reward
        total_steps += 1

        # learn from a random batch of past experience
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            s, a, r, s2, d = zip(*batch)
            s = torch.tensor(np.array(s))
            a = torch.tensor(a)
            r = torch.tensor(r, dtype=torch.float32)
            s2 = torch.tensor(np.array(s2))
            d = torch.tensor(d, dtype=torch.float32)

            q_pred = net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # Q of the action taken
            with torch.no_grad():
                q_next = target_net(s2).max(1).values
                target = r + gamma * q_next * (1 - d)  # no future after the episode ends

            loss = nn.functional.mse_loss(q_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_steps % target_update_every == 0:
            target_net.load_state_dict(net.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 200 == 0:
        print(f"Episode {episode+1}: last reward = {episode_reward}, epsilon = {epsilon:.2f}")

# ---------- SHOW WHAT IT LEARNED ----------
arrows = ["←", "↓", "→", "↑"]
print("\nLearned policy (G = goal, X = trap):")
for r in range(6):
    line = ""
    for c in range(6):
        if (r, c) == env.goal:
            line += "G "
        elif (r, c) in env.traps:
            line += "X "
        else:
            o = torch.tensor([r / 5, c / 5], dtype=torch.float32)
            line += arrows[net(o).argmax().item()] + " "
    print(line)

# ---------- TEST ----------
obs = env.reset()
done = False
path = [env.pos]
total = 0
while not done:
    with torch.no_grad():
        action = net(torch.tensor(obs)).argmax().item()
    obs, reward, done = env.step(action)
    path.append(env.pos)
    total += reward

print("\nPath taken:", path)
print("Total reward:", total, "(best possible is 1)")

# ---------- TEST FROM EVERY STARTING TILE ----------
ok = 0
total_tiles = 0
for r in range(6):
    for c in range(6):
        if (r, c) in env.traps or (r, c) == env.goal:
            continue
        total_tiles += 1
        env.reset()
        env.pos = (r, c)
        obs = env._obs()
        done = False
        reward = 0
        while not done:
            with torch.no_grad():
                action = net(torch.tensor(obs)).argmax().item()
            obs, reward, done = env.step(action)
        if env.pos == env.goal:
            ok += 1
        else:
            print(f"  failed from tile ({r}, {c})")

print(f"\nReached the goal from {ok}/{total_tiles} starting tiles")
