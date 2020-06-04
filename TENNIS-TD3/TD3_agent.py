import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from TDmodel import Actor, Critic
import copy
import torch.optim as optim

BUFFER_SIZE = int(1e6)   # REPLAY BUFFER SIZE
BATCH_SIZE = 100          # BATCH SAMPLED FROM REPLAY BUFFER
TAU = 5e-3                  # MERGE FACTOR
ACTOR_LR = 3e-4             # LR - ACTOR
CRITIC_LR = 3e-4            # LR - CRITIC
NUM_UPDATES = 1             # NUMBER OF UPDATES PER TIMESTEP
MIN_TRAJECTORY = 1000       # MIN NUMBER OF LEN OF REPLAY BUFFER BEFORE SAMPLING
POLICY_UPDATE = 2           # WHEN TO UPDATE THE POLICY
GAMMA = 0.99                # DISCOUNT FACTOR
# EPS = 1.0
# EPS_DECAY = 0.995
# UPDATE_NOISE = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)


        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr = CRITIC_LR)
        
        self.noise = OUNoise(action_size, seed)
        self.gauss_noise_act = Gaussian_Noise(0,0.1, action_size)
        self.policy_noise = 0.1
        # self.scale = EPS
        self.policy_clip = 0.5
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t = 0
        self

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise = True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            # action += self.noise.sample()
            action += self.gauss_noise_act.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done, update = False):
        self.memory.add(state, action, reward, next_state, done)
        self.t += 1
        if len(self.memory) > MIN_TRAJECTORY and update:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences



        noise = (torch.randn_like(actions)* self.policy_noise).clamp(-self.policy_clip, self.policy_clip)
        target_actions = torch.clamp(self.actor_target(next_states)+noise, -1, 1)

        Q_targets_next = torch.min(*self.critic_target(next_states, target_actions))

        Q_targets = rewards + gamma*(1-dones)*Q_targets_next

        # if self.t%UPDATE_NOISE==0 and self.t>10000:
        #     self.scale *= EPS_DECAY

        current_q1, current_q2 = self.critic_local(states, actions)
        critic_loss = F.mse_loss(current_q1, Q_targets) + F.mse_loss(current_q2, Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
        self.critic_optim.step()

        if self.t%POLICY_UPDATE==0:

            actor_loss = -self.critic_local.q1(states, self.actor_local(states)).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.soft_update(self.actor_local, self.actor_target, TAU)
            self.soft_update(self.critic_local, self.critic_target, TAU)
        


    
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)





class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Gaussian_Noise:
    def __init__(self, mu, std, size):
        self.mu = mu
        self.std = std
        self.size = size

    def sample(self):
        return np.random.normal(self.mu, self.std, self.size)
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

