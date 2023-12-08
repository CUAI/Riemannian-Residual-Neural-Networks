# Riemannian Residual Neural Networks

We provide the code for [Riemannian Residual Neural Networks](https://arxiv.org/abs/2006.10254) in this repository. 

Summary: We introduce a fully geometric and natural approach for learning residual networks over general Riemannian manifolds, given only the exponential map (i.e. only geodesic information). We demonstrate the utility of our approach by using it to improve hyperbolic graph learning and SPD video classification tasks.

We generalize the traditional Euclidean formula $x \leftarrow x + f_\theta(x)$ to the more general Riemannian expression $x \leftarrow \exp_x (f_\theta(x))$ via the Riemannian exponential map. Learning then takes place in the tangent space, i.e., we learn vector fields that induce our networks. A forward pass through a Riemannian Resnet defined over a single manifold is depicted below.

Riemannian ResNet Forward Pass on the Same Manifold            |
:-------------------------:
![RResNet Same Manifold](https://i.imgur.com/02vWOam.png)|

In cases were appropriate maps exist between manifolds, we can further generalize the forward pass, as depicted below.

Riemannian ResNet Forward Pass over Different Manifolds            |
:-------------------------:
![RResNet Different Manifolds](https://i.imgur.com/KP5sUdj.png)|

## Software Requirements

This codebase requires Python 3 and PyTorch 1.9+.

## Usage

### Installation

All main functionality is provided by the module `rresnet`. To install `rresnet`, clone the repository, and while in the root, run

```
pip install -e .
```

The `-e` option allows you to edit the files in rresnet without having to reinstall the package.

### Basic Example

Creating and using a Riemannian ResNet is simple. Please see the below example for the manifold of Euclidean space:

```python
import torch
from rresnet import RResNet, ProjVecField
import rresnet.manifolds.euclidean as euclidean

# set up regular Euclidean resnet
num_blocks = 10
resnet = RResNet(
    euclidean.Euclidean(), [ProjVecField(euclidean.Euclidean(), in_dim=2, hidden_dim=32, out_dim=2, n_hidden=2) for _ in range(num_blocks)]
)

out = resnet(torch.randn(10, 2))
out.sum().backward()
for p in resnet.parameters():
    print(p.grad)
```

Creating a Riemannian ResNet for hyperbolic space with the embedded vector field is just as simple (but the input is now required to be on manifold):

```python
import torch
from rresnet import RResNet, ProjVecField
import rresnet.manifolds.hyperbolic as hyperbolic

# set up a hyperbolic resnet
num_blocks = 10
hyp_resnet = RResNet(
    hyperbolic.Poincare(), [ProjVecField(hyperbolic.Poincare(), in_dim=2, hidden_dim=32, out_dim=2, n_hidden=2) for _ in range(num_blocks)]
)
```

### Application: Hyperbolic Graph Learning

One of the demonstrated applications in the paper is to hyperbolic graph learning. We build our model into the [Hyperbolic HGCN repo](https://github.com/HazyResearch/hgcn) as an encoder (please see the [Hyperbolic GCN paper](https://arxiv.org/abs/1910.12933) for full details on the hyperbolic graph learning tasks considered). In particular, this repo provides a modified version of the original Hyperbolic GCN repo, under the subfolder `hgcn`. Three models are added as encoders in [`hgcn/models/encoders.py`]():

1. `RRNetHyperbolic`, a Riemannian ResNet for hyperbolic space built using embedded vector fields

2. `RRNetHorosphereInduced`, a Riemannian ResNet for hyperbolic space built using horosphere projection-induced vector fields (please see our paper for full details)

3. `RRNetGraphHyperbolic`, which is 2 above, except with added graph information (we make use of the added graph via adjacency matrix multiplication, as is standard)

The final model is the most powerful version of our network. To execute a demo, `cd hgcn`, run `set_env.sh` (to setup the environment for the HGCN repo), and then use the following command to obtain `RRNetGraphHyperbolic`'s performance on Airport for the task of node classification (we run on CPU here for purposes of greater accessibility; if you have a GPU available, please run on GPU instead):

```
python train.py --act=lrelu --alpha=0.5 --bias=0 --cuda=-1 --dataset=airport --dim=8 --dropout=0.46119404634535327 --epochs=10000 --gamma=0.5 --grad-clip=None --hdim=16 --lr=0.08211442565478701 --lr-reduce-freq=700 --manifold=PoincareBall --model=RRNetGraphHyperbolic --num-horospheres=250 --momentum=0.846694085258576 --num-blocks=3 --num-layers=2 --optimizer=Adam --print-epoch=True --save=0 --save-dir=None --seed=1234 --task=nc --weight-decay=0.00011318436290086442 --normalize-feats=0 --patience=-1
```

We introduce the new parameter `num-horospheres` which specifies the number of learned horospherical feature projections that comprise the hyperbolic feature map of the Riemannian ResNet.

## Attribution

If you use this code or our results in your research, please cite:

```
@article{Katsman2023RiemannianRN,
  title={Riemannian Residual Neural Networks},
  author={Isay Katsman and Eric Ming Chen and Sidhanth Holalkere and Anna Asch and Aaron Lou and Ser-Nam Lim and Christopher De Sa},
  journal={ArXiv},
  year={2023},
  volume={abs/2310.10013},
  url={https://api.semanticscholar.org/CorpusID:264145900}
}
```