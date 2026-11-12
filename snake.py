import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class Snake:
    def __init__(self, size=8):
        self.size = size
        self.dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    def reset(self):
        mid = self.size // 2
        self.snake = [(mid, mid)]
        self.d = 1
        self.steps_since_food = 0
        self._place_food()
        return self._obs()

    def _place_food(self):
        free = [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.snake]
        self.food = random.choice(free)

    def _bad(self, pos):
        r, c = pos
        out = r < 0 or r >= self.size or c < 0 or c >= self.size
        return out or pos in self.snake

    def _space(self, action):
        # how many free cells the head can reach after this move (0 to 1)
        nd = (self.d + [0, 1, -1][action]) % 4
        hr, hc = self.snake[0]
        start = (hr + self.dirs[nd][0], hc + self.dirs[nd][1])
        if self._bad(start):
            return 0.0
        blocked = set(self.snake[:-1])  # the tail moves away, so it is not a wall
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in self.dirs:
                n = (r + dr, c + dc)
                if (0 <= n[0] < self.size and 0 <= n[1] < self.size
                        and n not in blocked and n not in seen):
                    seen.add(n)
                    stack.append(n)
        return len(seen) / (self.size * self.size)

    def _obs(self):
        hr, hc = self.snake[0]
        look = lambda d: (hr + self.dirs[d % 4][0], hc + self.dirs[d % 4][1])
        fr, fc = self.food
        return np.array([
            self._bad(look(self.d)),        # danger straight
            self._bad(look(self.d + 1)),    # danger right
            self._bad(look(self.d - 1)),    # danger left
            self.d == 0, self.d == 1, self.d == 2, self.d == 3,
            fr < hr, fc > hc, fr > hr, fc < hc,  # food up/right/down/left
            self._space(0), self._space(1), self._space(2),  # open space
        ], dtype=np.float32)

    def step(self, action):
        # action: 0 = straight, 1 = turn right, 2 = turn left
        self.d = (self.d + [0, 1, -1][action]) % 4
        hr, hc = self.snake[0]
        head = (hr + self.dirs[self.d][0], hc + self.dirs[self.d][1])
        self.steps_since_food += 1

        if self._bad(head):
            return self._obs(), -10, True, False  # died: real ending

        self.snake.insert(0, head)
        if head == self.food:
            reward = 10
            self.steps_since_food = 0
            if len(self.snake) == self.size * self.size:
                return self._obs(), reward, True, False  # board full: won
            self._place_food()
        else:
            self.snake.pop()
            reward = 0

        truncated = self.steps_since_food > 100  # wandering too long: just stop
        return self._obs(), reward, False, truncated


# ---------- NETWORK ----------
net = nn.Sequential(
    nn.Linear(14, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 3),
)
target_net = copy.deepcopy(net)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# ---------- SETTINGS ----------
env = Snake()
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 1500
batch_size = 64
target_update_every = 500
buffer = deque(maxlen=50000)
total_steps = 0
recent_scores = deque(maxlen=100)

# ---------- TRAINING ----------
for episode in range(episodes):
    obs = env.reset()
    over = False

    while not over:
        if random.random() < epsilon:
            action = random.randrange(3)
        else:
            with torch.no_grad():
                action = net(torch.tensor(obs)).argmax().item()

        next_obs, reward, terminated, truncated = env.step(action)
        over = terminated or truncated
        # only a REAL ending (death/win) means "no future"; a time cutoff does not
        buffer.append((obs, action, reward, next_obs, terminated))
        obs = next_obs
        total_steps += 1

        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            s, a, r, s2, d = zip(*batch)
            s = torch.tensor(np.array(s))
            a = torch.tensor(a, dtype=torch.int64)
            r = torch.tensor(r, dtype=torch.float32)
            s2 = torch.tensor(np.array(s2))
            d = torch.tensor(d, dtype=torch.float32)

            q_pred = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target = r + gamma * target_net(s2).max(1).values * (1 - d)
            loss = nn.functional.mse_loss(q_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_steps % target_update_every == 0:
            target_net.load_state_dict(net.state_dict())

    recent_scores.append(len(env.snake) - 1)  # food eaten this game
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: avg food last 100 games = "
              f"{np.mean(recent_scores):.2f}, epsilon = {epsilon:.2f}")

# ---------- TEST: 100 games, no exploring ----------
scores = []
for _ in range(100):
    obs = env.reset()
    over = False
    while not over:
        with torch.no_grad():
            action = net(torch.tensor(obs)).argmax().item()
        obs, reward, terminated, truncated = env.step(action)
        over = terminated or truncated
    scores.append(len(env.snake) - 1)

print(f"\nTest over 100 games: average food = {np.mean(scores):.1f}, best = {max(scores)}")

# ---------- SAVE + WATCH ONE GAME ----------
import time

torch.save(net.state_dict(), "snake.pt")

obs = env.reset()
over = False
while not over:
    grid = [["." for _ in range(env.size)] for _ in range(env.size)]
    grid[env.food[0]][env.food[1]] = "F"
    for i, (r, c) in enumerate(env.snake):
        grid[r][c] = "O" if i == 0 else "o"
    print("\033[2J\033[H" + "\n".join(" ".join(row) for row in grid))
    print(f"food eaten: {len(env.snake) - 1}")
    time.sleep(0.15)
    with torch.no_grad():
        action = net(torch.tensor(obs)).argmax().item()
    obs, reward, terminated, truncated = env.step(action)
    over = terminated or truncated
