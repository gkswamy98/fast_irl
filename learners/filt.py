import numpy as np
import gym
import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from torch.optim import Adam
import pybullet_envs
from oadam import OAdam

from nn_utils import linear_schedule, gradient_penalty
from buffer import ReplayBuffer, QReplayBuffer
from arch import Discriminator
from gym_wrappers import ResetWrapper, RewardWrapper, TremblingHandWrapper, GoalWrapper, AntMazeResetWrapper
from TD3_BC import TD3_BC
import d4rl
import os

class FILTER():
    def __init__(self, env):
        self.env = env

    def sample(
        self,
        env,
        policy,
        trajs,
        no_regret
        ):
        # rollout trajectories using a policy and add to replay buffer
        S_curr = []
        A_curr = []
        total_trajs = 0
        alpha = env.alpha
        env.alpha = 0
        s = 0
        while total_trajs < trajs:
            obs = env.reset()
            done = False
            while not done:
                S_curr.append(obs)
                act = policy.predict(obs)[0]
                A_curr.append(act)
                obs, _, done, _ = env.step(act)
                s += 1
                if done:
                    total_trajs += 1
                    break
        env.alpha = alpha
        if no_regret:
            self.replay_buffer.add(S_curr, A_curr)
        return torch.Tensor(S_curr), torch.Tensor(A_curr), s
    
    def train(self,
              expert_sa_pairs,
              expert_obs,
              expert_acts,
              a1, a2, a3,
              n_seed=0,
              n_exp=25,
              alpha=0.5,
              no_regret=False,
              device="cpu:0"):


        expert_sa_pairs = expert_sa_pairs.to(device)

        if alpha <= 1e-4:
            name = 'mm_icml'
        elif no_regret:
            name = 'filter_nr_icml'
        else:
            name = 'filter_br_icml'

        learn_rate = 8e-3
        batch_size = 4096
        f_steps = 1
        pi_steps = 5000
        num_traj_sample = 4
        outer_steps = 0
        mean_rewards = []
        std_rewards = []
        env_steps = []
        log_interval = 5
        
        cur_env = gym.make(self.env)
        if 'maze' in self.env:
            cur_env = AntMazeResetWrapper(GoalWrapper(cur_env), a1, a2, a3)
        else:
            cur_env = ResetWrapper(cur_env, a1, a2, a3, expert_obs, expert_acts)
       
        cur_env.alpha = alpha
        f_net = Discriminator(cur_env).to(device)
        f_opt = OAdam(f_net.parameters(), lr=learn_rate)
        cur_env = RewardWrapper(cur_env, f_net)
        
        if 'maze' in self.env:
            cur_env = TremblingHandWrapper(cur_env, p_tremble=0)
            eval_env = TremblingHandWrapper(GoalWrapper(gym.make(self.env)), p_tremble=0)

            state_dim = cur_env.observation_space.shape[0]
            action_dim = cur_env.action_space.shape[0] 
            max_action = float(cur_env.action_space.high[0])

            q_replay_buffer = QReplayBuffer(state_dim, action_dim)
            e = gym.make(self.env)
            dataset = e.get_dataset()
            q_dataset = d4rl.qlearning_dataset(e)
            q_replay_buffer.convert_D4RL(dataset, q_dataset)
            pi_replay_buffer = QReplayBuffer(state_dim, action_dim)

            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": 0.99,
                "tau": 0.005,
                # TD3
                "policy_noise": 0.2 * max_action,
                "noise_clip": 0.5 * max_action,
                "policy_freq": 2,
                # TD3 + BC
                "alpha": 2.5,
                "q_replay_buffer": q_replay_buffer,
                "pi_replay_buffer": pi_replay_buffer,
                "env": cur_env,
                "f": f_net,
            }
            pi = TD3_BC(**kwargs)
            for _ in range(1):
                pi.learn(total_timesteps=int(1e4), bc=True)
                mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=25)
                print(100 * mean_reward)
        else:
            cur_env = TremblingHandWrapper(cur_env, p_tremble=0.1)
            eval_env = TremblingHandWrapper(gym.make(self.env), p_tremble=0.1)
            pi = SAC('MlpPolicy', cur_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                        learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02, device=device)

            pi.actor.optimizer = OAdam(pi.actor.parameters())
            pi.critic.optimizer = OAdam(pi.critic.parameters())


        if no_regret:
            replay_buffer = ReplayBuffer(cur_env.observation_space.shape[0], cur_env.action_space.shape[0])
            self.replay_buffer = replay_buffer
        
        steps = 0
        for outer in range(outer_steps):
            if not outer == 0:
                learning_rate_used = learn_rate/outer
            else:
                learning_rate_used = learn_rate
            f_opt = OAdam(f_net.parameters(), lr=learning_rate_used)

            pi.learn(total_timesteps=pi_steps, log_interval=1000)
            steps += pi_steps

            S_curr, A_curr, s = self.sample(cur_env, pi, num_traj_sample, no_regret=no_regret)
            steps += s
            if no_regret: # use samples from the replay buffer
                tuple_samples = self.replay_buffer.sample(batch_size)
                obs_samples, act_samples = tuple_samples[0], tuple_samples[1]
                learner_sa_pairs = torch.cat((torch.tensor(obs_samples), torch.tensor(act_samples)), axis=1).to(device)
            else: # use samples from current policy
                learner_sa_pairs = torch.cat((S_curr, A_curr), dim=1).to(device)

            for _ in range(f_steps):
                learner_sa = learner_sa_pairs[np.random.choice(len(learner_sa_pairs), batch_size)]
                expert_sa = expert_sa_pairs[np.random.choice(len(expert_sa_pairs), batch_size)]
                f_opt.zero_grad()
                f_learner = f_net(learner_sa.float())
                f_expert = f_net(expert_sa.float())
                gp = gradient_penalty(learner_sa, expert_sa, f_net)
                loss = f_expert.mean() - f_learner.mean() + 10 * gp 
                loss.backward()
                f_opt.step()

            if outer % log_interval == 0:
                if 'maze' in self.env:
                    mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=25)
                    mean_reward = mean_reward * 100
                    std_reward = std_reward * 100
                else:
                    mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=10)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                env_steps.append(steps)
                print("{0} Iteration: {1}".format(int(outer), mean_reward))

            np.savez(os.path.join("learners", self.env, "{0}_rewards_{1}_{2}_{3}".format(name,
                                                                                         n_exp,
                                                                                         n_seed,
                                                                                         outer)),
                     means=mean_rewards, 
                     stds=std_rewards,
                     env_steps=env_steps)