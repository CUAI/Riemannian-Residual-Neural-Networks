from .manifold import Manifold


class Euclidean(Manifold):
    def __init__(self):
        super().__init__()

    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        return u

    def projx(self, x):
        return x

    def exp(self, x, u):
        return x + u

    def log(self, x, y):
        return y - x

    def __str__(self):
        return "Euclidean Space"
