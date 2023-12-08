import torch.nn as nn


def create_network(in_dim, hidden_dim, out_dim, n_hidden, act=nn.Tanh, opt_slope=0.5):
    if n_hidden == 0:
        return nn.Linear(in_dim, out_dim)
    network = [nn.Linear(in_dim, hidden_dim)]
    for _ in range(n_hidden - 1):
        network += [(act(negative_slope=opt_slope) if isinstance(act, nn.LeakyReLU) else act()), nn.Linear(hidden_dim, hidden_dim)]

    network += [(act(negative_slope=opt_slope) if isinstance(act, nn.LeakyReLU) else act()), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*network)
