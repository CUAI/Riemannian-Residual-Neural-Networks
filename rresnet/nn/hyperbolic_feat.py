import torch
from ..utils import EPS


def distance_to_horosphere(x, w, b):
    w = w / w.norm(dim=-1).unsqueeze(dim=-1)

    squeeze = False
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
        squeeze = True

    n = x.shape[1]
    num_horocycles = w.shape[0]

    x = x.unsqueeze(1).repeat(1, num_horocycles, 1)
    w = w.unsqueeze(0)
    b = b.unsqueeze(0)

    ret = 1 * torch.log((1 - torch.pow(x, 2).sum(dim=-1)) /
                        torch.pow(x - w, 2).sum(dim=-1).clamp_min(min=EPS[x.dtype])) + b

    return ret.squeeze() if squeeze else ret
