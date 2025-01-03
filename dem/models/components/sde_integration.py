from contextlib import contextmanager

import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.sdes import VEReverseSDE
from dem.utils.data_utils import remove_mean


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def grad_E(x, energy_function):
    with torch.enable_grad():
        x = x.requires_grad_()
        return torch.autograd.grad(torch.sum(energy_function(x)), x)[0].detach()


def negative_time_descent(x, energy_function, num_steps, dt=1e-4):
    samples = []
    for _ in range(num_steps):
        drift = grad_E(x, energy_function)
        x = x + drift * dt

        if energy_function.is_molecule:
            x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)

        samples.append(x)
    return torch.stack(samples)


def euler_maruyama_step(
    sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, diffusion_scale=1.0,
):

    # Calculate drift and diffusion terms
    drift = sde.f(t, x) * dt
    
    if sde.is_HPID:
        diffusion = diffusion_scale * np.sqrt(dt) * torch.randn_like(x)
    else:
        diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)
    
    # Update the state
    x_next = x + drift + diffusion
    return x_next, drift


def integrate_pfode(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool = False,
):
    start_time = 1.0 if reverse_time else 0.0
    end_time = 1.0 - start_time
    
    eps = 0.0 if reverse_time else 1e-6

    times = torch.linspace(start_time + eps, end_time - eps, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    with torch.no_grad():
        for t in times:
            x, f = euler_maruyama_step(sde, t, x, 1 / num_integration_steps)
            samples.append(x)

    return torch.stack(samples)


def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    negative_time=False,
    num_negative_time_steps=100,
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time
    
    eps = 0.0 if reverse_time else 1e-6

    times = torch.linspace(start_time + eps, end_time - eps, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []

    with conditional_no_grad(no_grad):
        for t in times:
            x, f = euler_maruyama_step(
                sde, t, x, time_range / num_integration_steps, diffusion_scale,
            )
            if energy_function.is_molecule:
                x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
            samples.append(x)

    samples = torch.stack(samples)
    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples

def backward_drift_HPID(
    x: torch.Tensor,
    times: torch.Tensor,
    beta: torch.Tensor,
):
    beta_sqrt = torch.sqrt(torch.tensor(beta, dtype=x.dtype, device=x.device))
    tanh_term = torch.tanh(beta_sqrt * times)
    
    return -(beta_sqrt / tanh_term) * x

def backward_integrate(
    x1: torch.Tensor,
    times: torch.Tensor,
    num_integration_steps: int,
    time_range=1.0,
):
    x = x1.clone()
    beta = 1.0
    dt = time_range / num_integration_steps
    
    for k in range(num_integration_steps):
        t_cur = time_range - k * dt
        mask = (times <= (k+1) * dt)
        
        if mask.any():
            drift = backward_drift_HPID(x[mask], t_cur, beta)
            
            noise = torch.randn(x[mask].shape, device=x.device)
            
            x[mask] = x[mask] + drift * dt + np.sqrt(dt) * noise
    
    return x