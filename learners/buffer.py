import numpy as np
import gym
import torch
import tqdm

class ReplayBuffer():
    def __init__(self, obs_space_size, action_space_size):
        self.obs_size = obs_space_size
        self.act_size = action_space_size
        self.size = 0
        self.obs = None
        self.actions = None
        self.first_addition = True

    def size(self):
        return self.size

    def add(self, obs, act):
        if not len(obs[0]) == self.obs_size or not len(act[0]) == self.act_size:
            raise Exception('incoming samples do not match the correct size')
        if self.first_addition:
            self.first_addition = False
            self.obs = np.array(obs)
            self.actions = np.array(act)
        else:
            self.obs = np.append(self.obs, np.array(obs), axis=0)
            self.actions = np.append(self.actions, np.array(act), axis=0)
        self.size += len(obs)
        return

    def sample(self, batch):
        indexes = np.random.choice(range(self.size), batch)
        return self.obs[indexes], self.actions[indexes]

class QReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def convert_D4RL(self, dataset, q_dataset, samps=int(1e6)):
        j = 0
        m = []
        for i in tqdm.tqdm(range(len(q_dataset['observations']))):
            while np.linalg.norm(q_dataset['observations'][i] - dataset['observations'][j]) > 1e-10:
                j += 1
            m.append(j)
        m = np.array(m) 
        goals = dataset['infos/goal'][m]

        j = 0
        m = []
        for i in tqdm.tqdm(range(len(q_dataset['next_observations']))):
            while np.linalg.norm(q_dataset['next_observations'][i] - dataset['observations'][j]) > 1e-10:
                j += 1
            m.append(j)
        m = np.array(m) 
        next_goals = dataset['infos/goal'][m]

        self.state = np.concatenate([q_dataset['observations'], goals], axis=1)[:samps]
        self.action = q_dataset['actions'][:samps]
        self.next_state = np.concatenate([q_dataset['next_observations'], next_goals], axis=1)[:samps]
        self.reward = q_dataset['rewards'].reshape(-1,1)[:samps]
        self.not_done = 1. - q_dataset['terminals'].reshape(-1,1)[:samps]
        self.size = self.state.shape[0]