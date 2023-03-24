import numpy as np
from torch import optim, nn
import torch
import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import pybullet_envs
import gym
import os
from gym_wrappers import TremblingHandWrapper, GoalWrapper

def BehavioralCloning(env, expert_obs, expert_acts, 
                      n_seed=0, device="cpu:0", pi=None,
                      steps=int(3e5)):
    
    if 'maze' not in env:
        expert_obs = torch.cat(expert_obs, dim=0)
        expert_acts =  torch.cat(expert_acts, dim=0)

    expert_obs = expert_obs.to(device)
    expert_acts = expert_acts.to(device)
    
    lr = 3e-4
    batch_size = 32
    loss_fn=nn.MSELoss()
    
    e = gym.make(env)
    if 'maze' in env:
        print("USE TD3+BC")
        exit()
    
    if pi is None:
        model = SAC('MlpPolicy', e, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), device=device)
        pi = model.policy.actor
        save_data=True
    else:
        save_data = False
    optimizer = optim.Adam(pi.parameters(), lr=lr)
    
    eval_e = gym.make(env)
    eval_e =  TremblingHandWrapper(env=eval_e, p_tremble=0.1)
    
    idxs = list(range(len(expert_obs)))

    for step in tqdm.tqdm(range(steps)):
        idx = np.random.choice(idxs, batch_size)
        states, actions = expert_obs[idx], expert_acts[idx]
        
        optimizer.zero_grad()
        outputs = pi(states.float())
        loss = loss_fn(outputs, actions.float())
        loss.backward()
        optimizer.step()
        

    mean_reward, std_reward = evaluate_policy(pi, eval_e, n_eval_episodes=10)
    print(n_seed, mean_reward)
    if save_data:
        np.savez(os.path.join("learners", env, "{0}_rewards_{1}".format('bc_icml', n_seed)),
                    means=mean_reward, 
                    stds=std_reward)
