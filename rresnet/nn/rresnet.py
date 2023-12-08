import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import create_network


class ProjVecField(nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim, n_hidden, act=nn.Tanh):
        super().__init__()
        self.func = create_network(in_dim, hidden_dim, out_dim, n_hidden, act=act)
        self.manifold = manifold

    def forward(self, x):
        v = self.func(x)
        return self.manifold.proju(x, v)


class FeatureMapVecFieldSimple(nn.Module):
    def __init__(self, fi, interm_dim, feat_dim, manifold):
        super().__init__()
        self.fi = fi
        self.interm_dim = interm_dim
        self.feat_dim = feat_dim

        self.interm = nn.Linear(feat_dim, interm_dim)
        self.bn1 = nn.BatchNorm1d(interm_dim)
        self.coeffs = nn.Linear(interm_dim, feat_dim)
        self.manifold = manifold

    def forward(self, x):
        feat = self.fi(x)
        interm = self.bn1(F.relu(self.interm(feat)))
        coeffs = self.coeffs(interm)  # bs x d
        new_vecs = torch.autograd.grad(
            self.fi(x), x, grad_outputs=coeffs, create_graph=True)
        ret = self.manifold.proju(x, new_vecs[0])
        return ret


class FeatureMapVecFieldSimpleGraphBase(nn.Module):
    def __init__(self, fi, interm_dim, feat_dim, manifold):
        super().__init__()
        self.fi = fi
        self.interm_dim = interm_dim
        self.feat_dim = feat_dim

        self.interm = nn.Linear(feat_dim, interm_dim)
        self.bn1 = nn.BatchNorm1d(interm_dim)
        self.coeffs = nn.Linear(interm_dim, feat_dim)
        self.manifold = manifold

    def forward(self, x, adj):
        feat = self.fi(x)
        feat = (adj @ (adj @ feat))
        interm = self.bn1(F.leaky_relu(self.interm(feat), negative_slope=0.5))
        coeffs = self.coeffs(interm)  # bs x d

        new_vecs = torch.autograd.grad(
            self.fi(x), x, grad_outputs=coeffs, create_graph=True)

        ret = self.manifold.proju(x, new_vecs[0])
        return ret


class RResNet(nn.Module):
    def __init__(self, manifold, vector_fields):
        """
        manifold (Manifold): Input Manifold
        vector_fields (nn.Module list): List of modules that map M -> TM
        """
        super().__init__()

        self.manifold = manifold
        self.vector_fields = nn.ModuleList(vector_fields)

    def forward(self, x, adj=None):
        for v_f in self.vector_fields:
            vf = v_f(x) if adj is None else v_f(x, adj)
            x = self.manifold.exp(x, vf)
            x = self.manifold.projx(x)

        return x
