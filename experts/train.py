import numpy as np
import gym
import pybullet_envs
from stable_baselines3 import SAC
from typing import Callable, Union
import argparse

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


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def get_state(env):
    p = env.env._p
    base_pos = [] # position and orientation of base for each body
    base_vel = [] # velocity of base for each body
    joint_states = [] # joint states for each body
    for i in range(p.getNumBodies()):
        base_pos.append(p.getBasePositionAndOrientation(i))
        base_vel.append(p.getBaseVelocity(i))
        joint_states.append([p.getJointState(i,j) for j in range(p.getNumJoints(i))])
    return base_pos, base_vel, joint_states

def rollout(pi, env, full_state=False):
    states = []
    actions = []
    if full_state:
        body_pos = []
        body_vel = []
        joint_states = []
    s = env.reset()
    if full_state:
        p, v, j = get_state(env.env)
        body_pos.append(p)
        body_vel.append(v)
        joint_states.append(j)
    done = False
    J = 0
    while not done:
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a)
        if full_state:
            p, v, j = get_state(env.env)
            body_pos.append(p)
            body_vel.append(v)
            joint_states.append(j)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    if full_state:
        return states, actions, J, body_pos, body_vel, joint_states
    else:
        return states, actions, J

def train_walker_expert():
    # No env normalization.
    env = gym.make('Walker2DBulletEnv-v0')
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=linear_schedule(7.3e-4), 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/Walker2DBulletEnv-v0/expert")

def train_hopper_expert():
    # No env normalization.
    env = gym.make('HopperBulletEnv-v0')
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=linear_schedule(7.3e-4), 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/HopperBulletEnv-v0/expert")

def train_halfcheetah_expert():
    # No env normalization.
    env = gym.make('HalfCheetahBulletEnv-v0')
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=7.3e-4, 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/HalfCheetahBulletEnv-v0/expert")

def generate_demos(env):
    model = SAC.load("experts/{0}/expert".format(env))
    def expert(s):
        return model.predict(s, deterministic=True)
    tot = 0
    demo_s = []
    demo_a = []
    P = []
    V = []
    C = []
    demo_env = TremblingHandWrapper(gym.make(env), p_tremble=0.1)
    for _ in range(100):
        s_traj, a_traj, J, p, v, c = rollout(expert, demo_env, full_state=True)
        demo_s.append(s_traj)
        demo_a.append(a_traj)
        P.append(p)
        V.append(v)
        C.append(c)
        tot += J
    print(tot / 100)
    # np.savez("experts/{0}/demos_tremble".format(env), s=demo_s, a=demo_a, P=P, V=V, C=C)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train expert policies.')
    parser.add_argument('env', choices=['walker', 'hopper', 'halfcheetah'])
    args = parser.parse_args()
    if args.env == 'walker':
        # train_walker_expert()
        generate_demos("Walker2DBulletEnv-v0")
    elif args.env == 'hopper':
        # train_hopper_expert()
        generate_demos("HopperBulletEnv-v0")
    elif args.env == 'halfcheetah':
        # train_halfcheetah_expert()
        generate_demos("HalfCheetahBulletEnv-v0")
    else:
        print("ERROR: unsupported env.")