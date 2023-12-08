import abc
import torch
import numpy as np


class Manifold(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def inner(self, x, u, v, keepdim=False):
        pass

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        return norm_sq if squared else norm_sq.sqrt()

    @abc.abstractmethod
    def proju(self, x, u):
        pass

    @abc.abstractmethod
    def projx(self, x):
        pass

    def egrad2rgrad(self, x, u):
        return self.proju(x, u)

    @abc.abstractmethod
    def exp(self, x, u):
        pass

    @abc.abstractmethod
    def log(self, x, y):
        pass

    def dist(self, x, y, squared=False, keepdim=False):
        return self.norm(x, self.log(x, y), squared, keepdim)

    @abc.abstractmethod
    def __str__(self):
        pass
