import numpy as np
import math
import torch
from torch.distributions.normal import Normal
from dem.utils.debug import fprint

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule


def wrap_for_richardsons(score_estimator):
    def _fxn(t, x, energy_function, noise_schedule, num_mc_samples):
        bigger_samples = score_estimator(t, x, energy_function, noise_schedule, num_mc_samples)

        smaller_samples = score_estimator(
            t, x, energy_function, noise_schedule, int(num_mc_samples / 2)
        )

        return (2 * bigger_samples) - smaller_samples

    return _fxn


def log_expectation_reward(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
):
    
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    h_t = noise_schedule.h(repeated_t).unsqueeze(1)

    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())

    log_rewards = energy_function(samples)

    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    return torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)

def estimate_grad_Rt(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
):
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))

    grad_fxn = torch.func.grad(log_expectation_reward, argnums=1)
    vmapped_fxn = torch.vmap(grad_fxn, in_dims=(0, 0, None, None, None), randomness="different")

    return vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples)

##########################################################################################################
# H-PID's optiaml control

def harmonic_IS(
    t: torch.Tensor,        # [batch_size]
    x: torch.Tensor,        # [batch_size, dim]
    energy_function: BaseEnergyFunction,
    num_mc_samples: int,
    beta: float,
):
    device = x.device
    dtype = x.dtype
    batch_size, dim = x.shape
    sqrt_beta = torch.sqrt(torch.tensor(beta, dtype=dtype, device=device))

    cosh_term = torch.cosh((1 - t) * sqrt_beta)
    sinh_term = torch.sinh((1 - t) * sqrt_beta)
    coth_term = 1 / torch.tanh(sqrt_beta)

    denominator = (cosh_term - sinh_term * coth_term).unsqueeze(-1)  # [batch_size, 1]
    denominator = denominator + 1e-8 * (denominator == 0).float()
    mean = x / denominator  # [batch_size, dim]

    diag_val = sqrt_beta * (
        1 / torch.tanh((1 - t) * sqrt_beta) - 1 / torch.tanh(sqrt_beta)
    )  # [batch_size]
    diag_val = diag_val + 1e-8 * (diag_val == 0).float()
    # Expand diag_val to [batch_size, dim] so diag_embed gives [batch_size, dim, dim]
    H = torch.diag_embed(diag_val.unsqueeze(-1).expand(batch_size, dim))  # [batch_size, dim, dim]
    # H = H + 1e-8 * torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)
    # H_inv = torch.inverse(H)  # [batch_size, dim, dim]
    H_inv = torch.diag_embed((1 / diag_val).unsqueeze(-1).expand(batch_size, dim))          # [batch_size, dim, dim]

    # Ensure positive-definiteness
    # epsilon = 1e-8
    # H_inv = H_inv + 1e-8 * torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)  # [batch_size, dim, dim]

    # Sample from MVN
    z = torch.randn(batch_size, num_mc_samples, dim, dtype=dtype, device=device)
    L = torch.linalg.cholesky(H_inv)  # [batch_size, dim, dim]
    y = mean.unsqueeze(1) + torch.matmul(z, L)  # [batch_size, num_mc_samples, dim]
    
    # if any(tensor.isnan().sum() > 0 for tensor in [denominator, mean, diag_val, H_inv, H, y]):
    #     fprint(f"nan_check : denominator, {denominator.isnan().sum()}")
    #     fprint(f"nan_check : mean, {mean.isnan().sum()}")
    #     fprint(f"nan_check : diag_val, {diag_val.isnan().sum()}")
    #     fprint(f"nan_check : H_inv, {H_inv.isnan().sum()}")
    #     fprint(f"nan_check : H, {H.isnan().sum()}")
    #     fprint(f"nan_check : y, {y.isnan().sum()}")
    #     print("x :", x)
    #     print("denominator :", denominator)
    #     raise ValueError("제발 왜그러는거야")
    
    # if any(tensor.isinf().sum() > 0 for tensor in [denominator, mean, diag_val, H_inv, H, y]):
    #     fprint(f"inf_check : denominator, {denominator.isinf().sum()}")
    #     fprint(f"inf_check : mean, {mean.isinf().sum()}")
    #     fprint(f"inf_check : diag_val, {diag_val.isinf().sum()}")
    #     fprint(f"inf_check : H_inv, {H_inv.isinf().sum()}")
    #     fprint(f"inf_check : H, {H.isinf().sum()}")
    #     fprint(f"inf_check : y, {y.isinf().sum()}")
    #     print("x :", x)
    #     print("denominator :", denominator)
    #     raise ValueError("제발 왜그러는거야")

    return H, mean, y

def compute_control(
    t: torch.Tensor,         # [batch_size]
    x: torch.Tensor,         # [batch_size, dim]
    y: torch.Tensor,         # [batch_size, num_mc_samples, dim]
    y_star: torch.Tensor,    # [batch_size, dim]
    H_inv: torch.Tensor,     # [batch_size, dim, dim]
    energy_function: BaseEnergyFunction,
    num_mc_samples: int,
    beta: float,
    clipper: Clipper = None,
):
    device = x.device
    dtype = x.dtype
    beta_sqrt = torch.sqrt(torch.tensor(beta, dtype=dtype, device=device))

    cosh_term = torch.cosh((1 - t) * beta_sqrt)
    sinh_term = torch.sinh((1 - t) * beta_sqrt)
    coth_beta = 1 / torch.tanh(beta_sqrt)

    diff = y - y_star.unsqueeze(1)
    mahalanobis = 0.5 * torch.einsum('bnd,bdd,bnd->bn', diff, H_inv, diff)

    log_rewards = energy_function(y)
    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    x_squared = torch.sum(x**2, dim=-1, keepdim=True) # Shape: (batch_size_x, 1)
    y_squared = torch.sum(y**2, dim=-1) # Shape: (batch_size_y)
    inner_product = torch.sum(x.unsqueeze(1)*y, dim=-1)
    # x_squared = torch.sum(x ** 2, dim=-1, keepdim=True)  # Shape: (batch_size_x, 1)
    # y_squared = torch.sum(y ** 2, dim=-1)  # Shape: (batch_size_y)
    # x_expanded = x.unsqueeze(1)  # Shape: (batch_size_x, 1, dim)
    # y_expanded = y.unsqueeze(0)  # Shape: (1, batch_size_y, dim)
    # inner_product = torch.sum(x_expanded * y_expanded, dim=-1)  # Shape: (batch_size_x, batch_size_y)


    log_G_ratio = (-beta_sqrt * (cosh_term.unsqueeze(-1)*(x_squared + y_squared) - 2*inner_product)
                   / (2 * sinh_term.unsqueeze(-1))) + (y_squared * beta_sqrt / 2) * coth_beta

    exponent = log_rewards + log_G_ratio - mahalanobis # Shape: (batch_size_x, batch_size_y)
    w = torch.softmax(exponent, dim=1) # Shape: (batch_size_x, batch_size_y)

    weighted_sum = torch.sum(w.unsqueeze(-1)*y, dim=1) # Shape: (batch_size_x, dim)
    u = (weighted_sum - x * cosh_term.unsqueeze(-1)) * (beta_sqrt / sinh_term.unsqueeze(-1)) # Shape: (batch_size_x, dim)
    
    if any(tensor.isnan().sum() > 0 for tensor in [mahalanobis, log_G_ratio, exponent, w, u]):
        fprint(f"nan_check : mahalanobis, {mahalanobis.isnan().sum()}")
        fprint(f"nan_check : log_G_ratio, {log_G_ratio.isnan().sum()}")
        fprint(f"nan_check : exponent, {exponent.isnan().sum()}")
        fprint(f"nan_check : w, {w.isnan().sum()}")
        fprint(f"nan_check : u, {u.isnan().sum()}")
        raise ValueError("제발 왜그러는거야")
    
    
    if any(tensor.isnan().sum() > 0 for tensor in [mahalanobis, log_G_ratio, exponent, w, u]):
        fprint(f"inf_check : mahalanobis, {mahalanobis.isinf().sum()}")
        fprint(f"inf_check : log_G_ratio, {log_G_ratio.isinf().sum()}")
        fprint(f"inf_check : exponent, {exponent.isinf().sum()}")
        fprint(f"inf_check : w, {w.isinf().sum()}")
        fprint(f"inf_check : u, {u.isinf().sum()}")
        raise ValueError("제발 왜그러는거야")
    
        
    return u

def harmonic_integral(
    t: torch.Tensor,        # [batch_size]
    x: torch.Tensor,        # [batch_size, dim]
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    beta: float = 0.1,
    clipper: Clipper = None,
):
    
    H_inv, y_star, y = harmonic_IS(t, x, energy_function, num_mc_samples, beta)
    u = compute_control(t, x, y, y_star, H_inv, energy_function, num_mc_samples, beta, clipper)
    
    return u
