import torch


def distance_to_hyperplane(x, w, b):
    squeeze = False
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
        squeeze = True

    num_hyperplanes = w.shape[0]

    x = x.unsqueeze(1).repeat(1, num_hyperplanes, 1)
    w = w.unsqueeze(0)
    b = b.unsqueeze(0)

    ret = torch.abs((x * w).sum(dim=-1) + b)/ w.norm(dim=-1)

    return ret.squeeze() if squeeze else ret
 