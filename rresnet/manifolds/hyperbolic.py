import torch
from .manifold import Manifold
from ..utils import EPS, tanh, artanh


class Poincare(Manifold):
    def __init__(self):
        super().__init__()
        self.edge_eps = 1e-3

    def lambda_x(self, x, keepdim=False):
        return 2 / (1 - x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])

    def inner(self, x, u, v, keepdim=False):
        return self.lambda_x(x, keepdim=True) ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        return u

    def projx(self, x, inplace=False):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
        maxnorm = (1 - self.edge_eps)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def exp(self, x, u):
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype])
        second_term = tanh(0.5 * self.lambda_x(x, keepdim=True) * u_norm) * u/(u_norm)
        gamma_1 = self.mobius_addition(x, second_term)
        return gamma_1

    def exp0(self, u):
        return self.exp(torch.zeros_like(u), u)

    def mobius_addition(self, x, y):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        denom = 1 + 2 * xy + x2 * y2
        return num / denom.clamp_min(EPS[x.dtype])

    def log(self, x, y):
        sub = self.mobius_addition(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        lam = self.lambda_x(x, keepdim=True)
        return 2 / lam * artanh(sub_norm) * sub / sub_norm

    def log0(self, y):
        return self.log(torch.zeros_like(y), y)

    def rand(self, *shape, out=None):
        x = torch.randn(*shape)
        return self.projx(x)

    def randvec(self, x, norm=1):
        y = torch.rand(x.shape)
        return y * norm / y.norm(dim=-1, keepdim=True)

    def __str__(self):
        return "Hyperbolic Space (Poincare)"
