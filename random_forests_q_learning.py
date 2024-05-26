import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REPLAY MEMORY
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Random Forest Q-Learning Model
class RandomForestQ:
    def __init__(self, n_estimators=10):
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.is_trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        if self.is_trained:
            return self.model.predict(X)
        else:
            return np.zeros((X.shape[0], 2))

# TRAINING PARAMETERS
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = RandomForestQ()
target_net = RandomForestQ()

memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state_np = state.cpu().numpy()
            q_values = policy_net.predict(state_np)
            return torch.tensor([[np.argmax(q_values)]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net.predict(state_batch.cpu().numpy())

    next_state_values = np.zeros(BATCH_SIZE)
    if non_final_next_states.shape[0] > 0:
        next_state_values[non_final_mask] = target_net.predict(non_final_next_states.cpu().numpy()).max(1)

    expected_state_action_values = reward_batch.cpu().numpy() + (GAMMA * next_state_values)

    X = state_batch.cpu().numpy()
    y = expected_state_action_values

    policy_net.fit(X, y)

# Main training loop
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net.model = policy_net.model

        # Render the environment to visualize the agent
        env.render()

        if done:
            break

print('Complete')
env.close()
