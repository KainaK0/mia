import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing
from torch.utils.tensorboard import SummaryWriter  # opcional para logs

# ------------------- Hiperparámetros ------------------- #
learning_rate = 1e-4
gamma = 0.99
batch_size = 32
# 1) Disminuir el tamaño del buffer a 20,000
buffer_size = 20_000

update_every = 4       # entrenar cada 4 pasos
target_update = 10_000 # copiado duro cada 10k steps
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 1e-6

num_episodes = 5000
max_steps = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear entorno con wrappers
env = gym.make("ALE/MsPacman-v5", full_action_space=False, frameskip=1)
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4,
                         scale_obs=False)
env = FrameStackObservation(env, stack_size=4)
action_size = env.action_space.n

# Definir red neuronal (Double DQN)
class QNetwork(nn.Module):
    def __init__(self, action_size, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        # Como tenemos 4 frames en gris: in_channels = 1 * 4 = 4
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = x / 255.0  # Normalización [0,1] para imágenes en uint8
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        # 2) Guardar estados como uint8 para reducir uso de memoria
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convertir a tensores float aquí (al extraer el batch)
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

# Agente con Double DQN
class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.local_q = QNetwork(action_size).to(device)
        self.target_q = QNetwork(action_size).to(device)
        # Sincronización inicial
        self.target_q.load_state_dict(self.local_q.state_dict())
        self.target_q.eval()
        
        self.optimizer = optim.Adam(self.local_q.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.t_step = 0  # Para controlar cada cuántos steps entrenamos
        self.total_steps = 0  # Para controlar el target_update
        
    def step(self, state, action, reward, next_state, done):
        # Guardar en buffer
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        self.total_steps += 1
        
        # Entrenar la red cada 'update_every' pasos
        if self.t_step % update_every == 0 and len(self.memory) > batch_size:
            experiences = self.memory.sample(batch_size)
            self.learn(experiences, gamma)
        
        # Hard update de la red target cada 'target_update' steps
        if self.total_steps % target_update == 0:
            self.target_q.load_state_dict(self.local_q.state_dict())
    
    def act(self, state, eps=0.0):
        """state es un np.array con shape (4, 84, 84)"""
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)
        else:
            state_t = torch.from_numpy(np.array([state])).float().to(device)
            with torch.no_grad():
                q_values = self.local_q(state_t)
            return int(q_values.argmax(dim=1).item())
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # 1) Seleccionar acción con la red local
        next_actions = self.local_q(next_states).argmax(dim=1, keepdim=True)
        
        # 2) Evaluar esa acción con la red target
        Q_targets_next = self.target_q(next_states).gather(1, next_actions)
        
        # 3) Calcular objetivo
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # 4) Q esperado de la red local
        Q_expected = self.local_q(states).gather(1, actions)
        
        # 5) Calcular pérdida
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # 6) Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = DQNAgent(action_size)

scores = []
eps = epsilon_start

for e in range(1, num_episodes+1):
    # Reiniciar el entorno
    state, _ = env.reset()
    state = np.array(state)  # (4, 84, 84)
    episode_score = 0
    
    for t in range(max_steps):
        action = agent.act(state, eps)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state)
        
        agent.step(state, action, reward, next_state, done)
        state = next_state
        episode_score += reward
        
        if done:
            break
    
    # Epsilon decay
    eps = max(epsilon_end, eps - epsilon_decay)
    scores.append(episode_score)
    
    print(f"\rEpisode {e} | Score: {episode_score} | Epsilon: {eps:.5f}", end="")
    
    if e % 100 == 0:
        avg_100 = np.mean(scores[-100:])
        print(f"\rEpisode {e}, Average Score (últ. 100): {avg_100:.2f}")

env.close()
torch.save(agent.local_q.state_dict(), "model_final_v4.pth")
