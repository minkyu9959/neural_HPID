import torch
from dem.models.components.score_estimator import harmonic_integral
_EPSILON = 1e-6


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, diffusion):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        return self.drift(t, x)

    def g(self, t, x):
        return self.diffusion(t, x)


class VEReverseSDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, score, noise_schedule, energy_function, is_HPID=False):
        super().__init__()
        self.score = score
        self.noise_schedule = noise_schedule
        self.energy_function = energy_function
        self.is_HPID = is_HPID

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)
            
        if self.is_HPID:
            # HPID's (forward) drift
            score = self.score(t, x)
            # score = harmonic_integral(t, x, self.energy_function, self.noise_schedule, 100)
        else:
            # iDEM's (reverse) drift
            score = self.g(t, x).pow(2) * self.score(t, x)

        return score

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        return g.unsqueeze(1) if g.ndim > 0 else torch.full_like(x, g)


class RegVEReverseSDE(VEReverseSDE):
    def f(self, t, x):
        dx = super().f(t, x[..., :-1])
        quad_reg = 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)
        return torch.cat([dx, quad_reg], dim=-1)

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        if g.ndim > 0:
            return g.unsqueeze(1)
        return torch.cat([torch.full_like(x[..., :-1], g), torch.zeros_like(x[..., -1:])], dim=-1)
