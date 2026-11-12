import gymnasium as gym
import numpy as np

# ---------- 1. SETUP ----------
env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

# ---------- 2. HYPERPARAMETERS ----------
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999
episodes = 5000

wins = 0

# ---------- 3. TRAINING LOOP ----------
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        best_next = np.max(Q[next_state])
        target = reward + gamma * best_next
        error = target - Q[state, action]
        Q[state, action] += alpha * error

        state = next_state

    wins += reward
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode+1}: wins in last 500 = {wins:.0f}, epsilon = {epsilon:.2f}")
        wins = 0

# ---------- 4. LOOK AT WHAT IT LEARNED ----------
print("\nLearned Q-table (rows = states, cols = left/down/right/up):")
print(np.round(Q, 3))

arrows = ["←", "↓", "→", "↑"]
print("\nLearned policy:")
for row in range(4):
    line = ""
    for col in range(4):
        s = row * 4 + col
        line += arrows[np.argmax(Q[s])] + " "
    print(line)

# ---------- 5. TEST THE TRAINED AGENT ----------
test_wins = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    test_wins += reward

print(f"\nTest: reached the goal {test_wins:.0f}/100 times")
