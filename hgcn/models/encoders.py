"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath

from rresnet import RResNet, ProjVecField, FeatureMapVecFieldSimple, FeatureMapVecFieldSimpleGraphBase
import rresnet.manifolds.hyperbolic as hyperbolic
from rresnet.nn.hyperbolic_feat import distance_to_horosphere


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)


class RRNetHyperbolic(Encoder):
    """
    Riemannian ResNet over hyperbolic space with an embedded vector field
    """

    def __init__(self, c, args):
        super(RRNetHyperbolic, self).__init__(c)
        self.manifold = hyperbolic.Poincare()

        self.linear = nn.Linear(args.feat_dim, args.dim)

        acts = {
            'relu': nn.ReLU(),
            None: nn.Identity(),
        }

        self.act = acts.get(args.act, nn.Identity)

        self.rresnet = RResNet(
            hyperbolic.Poincare(), [ProjVecField(
                hyperbolic.Poincare(), args.dim, args.hdim, args.dim, args.num_layers, act=self.act) for _ in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        return self.act(self.rresnet(self.manifold.projx(self.linear(x))))


class RRNetHorosphereInduced(Encoder):
    """
    Riemannian ResNet with horosphere-induced vector fields
    """

    def __init__(self, c, args, learnable_horospheres=True):
        super(RRNetHorosphereInduced, self).__init__(c)

        self.manifold = hyperbolic.Poincare()
        self.linear = nn.Linear(args.feat_dim, args.dim)

        self.learnable_horospheres = learnable_horospheres

        acts = {
            'relu': nn.ReLU,
            None: nn.Identity,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.num_horospheres = args.num_horospheres

        self.w = nn.Parameter(torch.randn(args.num_blocks, self.num_horospheres,
                              args.dim, device=args.device), requires_grad=learnable_horospheres)

        self.b = nn.Parameter(torch.randn(args.num_blocks, self.num_horospheres,
                              device=args.device), requires_grad=learnable_horospheres)

        f = [lambda x: distance_to_horosphere(
            x, self.w[i], self.b[i]) for i in range(args.num_blocks)]

        self.rresnet = RResNet(
            hyperbolic.Poincare(), [FeatureMapVecFieldSimple(
                f[i], args.hdim, self.num_horospheres, hyperbolic.Poincare()) for i in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        x = self.linear(x)
        x = self.manifold.projx(x)
        x = self.rresnet(x)
        x = self.act()(x)
        return x


class RRNetGraphHyperbolic(Encoder):
    """
    Riemannian ResNet with graph information
    """

    def __init__(self, c, args, learnable_horospheres=True):
        super(RRNetGraphHyperbolic, self).__init__(c)

        self.manifold = hyperbolic.Poincare()
        self.linear = nn.Linear(args.feat_dim, args.dim)

        self.learnable_horospheres = learnable_horospheres

        acts = {
            'relu': nn.ReLU,
            None: nn.Identity,
            'lrelu': nn.LeakyReLU,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.num_horospheres = args.num_horospheres

        self.w = nn.Parameter(torch.randn(args.num_blocks, self.num_horospheres,
                              args.dim, device=args.device), requires_grad=learnable_horospheres)

        self.b = nn.Parameter(torch.randn(args.num_blocks, self.num_horospheres,
                              device=args.device), requires_grad=learnable_horospheres)

        f = [lambda x: distance_to_horosphere(
            x, self.w[i], self.b[i]) for i in range(args.num_blocks)]

        self.rresnet = RResNet(
            hyperbolic.Poincare(), [FeatureMapVecFieldSimpleGraphBase(
                f[i], args.hdim, self.num_horospheres, hyperbolic.Poincare()) for i in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        x = self.linear(x)
        x = self.manifold.projx(x)
        x = self.rresnet(x, adj)
        x = self.act(negative_slope=0.5)(x)
        return x


class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)
