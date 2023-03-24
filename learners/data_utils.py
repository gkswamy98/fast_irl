import numpy as np
import torch
import d4rl
import gym

def fetch_demos(env):
    if 'maze' not in env:
        demos = np.load("experts/{0}/demos_full_2.npz".format(env), allow_pickle=True)
        demo_s = demos["s"]
        demo_a = demos["a"]
        P = demos["P"]
        V = demos["V"]
        C = demos["C"]

        expert_obs, expert_acts = tuple([torch.tensor(x) for x in demo_s]), tuple([torch.tensor(x) for x in demo_a])
        expert_sa_pairs = torch.cat((torch.cat(expert_obs), torch.cat(expert_acts)), dim=1)

        return expert_sa_pairs, expert_obs, expert_acts, P, V, C
    else:
        e = gym.make(env)
        dataset = e.get_dataset()
        term = np.argwhere(np.logical_or(dataset['timeouts'] > 0, dataset['terminals'] > 0))
        Js = []
        ranges = []
        start = 0
        for i in range(len(term)):
            ranges.append((start, term[i][0] + 1))
            J = dataset['rewards'][start: term[i][0] + 1].sum()
            Js.append(J)
            start = term[i][0] + 1
        Js = np.array(Js)
        exp_ranges = np.array(ranges)
        acts = np.concatenate([dataset['actions'][exp_range[0]:exp_range[1]] for exp_range in exp_ranges])
        obs = np.concatenate([dataset['observations'][exp_range[0]:exp_range[1]] for exp_range in exp_ranges])
        goals = np.concatenate([dataset['infos/goal'][exp_range[0]:exp_range[1]] for exp_range in exp_ranges])
        expert_obs = torch.cat([torch.tensor(obs), torch.tensor(goals)], dim=1)
        expert_acts = torch.tensor(acts)
        expert_sa_pairs = torch.cat((expert_obs, expert_acts), dim=1)
        qpos = np.array([dataset['infos/qpos'][exp_range[0]:exp_range[1]] for exp_range in exp_ranges])
        qvel = np.array([dataset['infos/qvel'][exp_range[0]:exp_range[1]] for exp_range in exp_ranges])
        G = np.array([dataset['infos/goal'][exp_range[0]:exp_range[1]] for exp_range in exp_ranges])
        return expert_sa_pairs, expert_obs, expert_acts, qpos, qvel, G