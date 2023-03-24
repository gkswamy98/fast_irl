import numpy as np
import torch
import gym
import os
import sys

def set_state(env, base_pos, base_vel, joint_states):
    p = env.env._p
    for i in range(p.getNumBodies()):
        p.resetBasePositionAndOrientation(i,*base_pos[i])
        p.resetBaseVelocity(i,*base_vel[i])
        for j in range(p.getNumJoints(i)):
            p.resetJointState(i,j,*joint_states[i][j][:2])

class ResetWrapper(gym.Wrapper):
    def __init__(self, env, P, V, C, expert_obs, expert_acts):
        super().__init__(env)
        self.env = env
        self.alpha = 0.5
        self.P = P
        self.V = V
        self.C = C
        self.expert_obs = expert_obs
        self.expert_acts = expert_acts
        self.t = 0
        self.max_t = 1000

    def reset(self):
        self.env.reset()
        if np.random.uniform() < self.alpha:
            idx = np.random.choice(len(self.P))
            t = np.random.choice(min(len(self.P[idx]), self.max_t))
            set_state(self.env, self.P[idx][0], self.V[idx][0], self.C[idx][0])
            obs = self.env.env.robot.calc_state()
            for i in range(t):
                a = self.expert_acts[idx][i]
                next_obs, rew, done, _ = self.env.step(a)
                obs = next_obs
            self.t = t
        else:
            self.t = 0
        return self.env.env.robot.calc_state()

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.max_t:
            done = True
        return next_obs, rew, done, info

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, function):
        super().__init__(env)
        self.env = env
        self.cur_state = None
        self.function = function
        self.low = env.action_space.low
        self.high = env.action_space.high

    def reset(self):
        obs = self.env.reset()
        self.cur_state = obs
        return obs

    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        #combine action and state
        sa_pair = np.concatenate((self.cur_state, action))
        reward = -(self.function.forward(torch.tensor(sa_pair, dtype=torch.float).to("cuda")))
        self.cur_state = next_state

        return next_state, reward, done, info

class TremblingHandWrapper(gym.Wrapper):
    def __init__(self, env, p_tremble=0.01):
        super().__init__(env)
        self.env = env
        self.p_tremble = p_tremble

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform() < self.p_tremble:
            action = self.env.action_space.sample()
        return self.env.step(action)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,))

    def reset(self):
        with HiddenPrints():
            obs = self.env.reset()
            goal = self.env.target_goal
            return np.concatenate([obs, goal])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        goal = self.env.target_goal
        return np.concatenate([obs, goal]), rew, done, info

class AntMazeResetWrapper(gym.Wrapper):
    def __init__(self, env, qpos, qvel, G):
        super().__init__(env)
        self.env = env
        self.alpha = 1
        self.qpos = qpos
        self.qvel = qvel
        self.G = G
        self.t = 0
        self.T = 700

    def reset(self):
        obs = self.env.reset()
        if np.random.uniform() < self.alpha:
            idx = np.random.choice(len(self.qpos))
            t = np.random.choice(len(self.qpos[idx]))
            with HiddenPrints():
                self.env.set_target(tuple(self.G[idx][t]))
            self.env.set_state(self.qpos[idx][t], self.qvel[idx][t])
            self.t = t
            obs = self.env.env.wrapped_env._get_obs()
            goal = self.env.target_goal
            obs = np.concatenate([obs, goal])
        else:
            self.t = 0
        return obs

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.T:
            done = True
        return next_obs, rew, done, info