import numpy as np
import torch
from torch import nn
from typing import List, Type
from typing import Callable, Union, Type, Optional, Dict, Any
from torch.autograd import Variable
from itertools import repeat
from torch.autograd import grad as torch_grad

def gradient_penalty(learner_sa, expert_sa, f, device="cuda"):
    batch_size = expert_sa.size()[0]

    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(expert_sa)

    interpolated = alpha * expert_sa.data + (1 - alpha) * learner_sa.data

    interpolated = Variable(interpolated, requires_grad=True).to(device)

    f_interpolated = f(interpolated.float()).to(device)

    gradients = torch_grad(outputs=f_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(f_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0].to(device)

    gradients = gradients.view(batch_size, -1)
    norm = gradients.norm(2, dim=1).mean().item()

    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    # 2 * |f'(x_0)|
    return ((gradients_norm - 0.4) ** 2).mean()

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

def init_ortho(layer):
    if type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight)

def create_mlp(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    return modules