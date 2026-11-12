import numpy as np


class GridWorld:
    """6x6 grid. Start top-left, goal bottom-right, traps in between."""

    def __init__(self):
        self.size = 6
        self.goal = (5, 5)
        self.traps = {(1, 1), (2, 3), (3, 1), (4, 4), (1, 4)}
        self.max_steps = 50
        self.n_states = self.size * self.size
        self.n_actions = 4  # 0=left, 1=down, 2=right, 3=up

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return self._state()

    def _state(self):
        # turn (row, col) into a single number 0-35
        return self.pos[0] * self.size + self.pos[1]

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

        # walls: stay inside the grid
        r = min(max(r, 0), self.size - 1)
        c = min(max(c, 0), self.size - 1)
        self.pos = (r, c)
        self.steps += 1

        if self.pos == self.goal:
            return self._state(), 10, True
        if self.pos in self.traps:
            return self._state(), -10, True
        if self.steps >= self.max_steps:
            return self._state(), -1, True
        return self._state(), -1, False


# ---------- TRAINING ----------
env = GridWorld()
Q = np.zeros((env.n_states, env.n_actions))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999
episodes = 5000

for episode in range(episodes):
    state = env.reset()
    done = False
    total = 0

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = env.step(action)

        target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state
        total += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode+1}: last reward = {total}, epsilon = {epsilon:.2f}")

# ---------- SHOW WHAT IT LEARNED ----------
arrows = ["←", "↓", "→", "↑"]
print("\nLearned policy (G = goal, X = trap):")
for r in range(env.size):
    line = ""
    for c in range(env.size):
        if (r, c) == env.goal:
            line += "G "
        elif (r, c) in env.traps:
            line += "X "
        else:
            line += arrows[np.argmax(Q[r * env.size + c])] + " "
    print(line)

# ---------- TEST ----------
state = env.reset()
done = False
path = [env.pos]
total = 0
while not done:
    state, reward, done = env.step(np.argmax(Q[state]))
    path.append(env.pos)
    total += reward

print("\nPath taken:", path)
print("Total reward:", total, "(best possible is 1)")
