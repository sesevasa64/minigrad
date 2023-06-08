import random
from enum import auto, Enum
from numbers import Real
from typing import List, Self


class GradFn(Enum):
    Add = auto()
    Sub = auto()
    Mul = auto()
    Div = auto()
    Pow = auto()


class Tensor:
    def __init__(self, value: Real, parents: List[Self] = [], requires_grad = False, grad_fn = None):
        assert isinstance(value, Real)
        self.value = value
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.parents = parents
        self.grad = 0
    def backward(self, grad=1):
        assert self.requires_grad
        if not self.parents:
            self.grad += grad
        for i, p in enumerate(self.parents):
            if not p.requires_grad:
                continue
            grad_new = grad
            match self.grad_fn:
                case GradFn.Add:
                    pass
                case GradFn.Sub if i == 1:
                    grad_new *= -1
                case GradFn.Mul:
                    m = self.parents[1-i].value
                    grad_new *= m
                case GradFn.Div if i == 1:
                    m = self.parents[0].value
                    grad_new *= -m / (p.value ** 2)
                case GradFn.Pow if i == 0:
                    m = self.parents[1].value
                    grad_new *= m * (p.value ** (m - 1))
            p.backward(grad_new)
    @classmethod
    def uniform(cls, low, high, requires_grad = False):
        return cls(random.uniform(low, high), requires_grad=requires_grad)
    def __add__(self, other: Self):
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(
            self.value + other.value, parents=[self, other], 
            requires_grad=requires_grad, grad_fn=GradFn.Add
        )
    def __sub__(self, other: Self):
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(
            self.value - other.value, parents=[self, other], 
            requires_grad=requires_grad, grad_fn=GradFn.Sub
        )
    def __mul__(self, other: Self):
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(
            self.value * other.value, parents=[self, other], 
            requires_grad=requires_grad, grad_fn=GradFn.Mul
        )
    def __truediv__(self, other: Self):
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(
            self.value / other.value, parents=[self, other], 
            requires_grad=requires_grad, grad_fn=GradFn.Div
        )
    def __pow__(self, other: Self):
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(
            self.value ** other.value, parents=[self, other], 
            requires_grad=requires_grad, grad_fn=GradFn.Pow
        )
    def __eq__(self, other: Self):
        return self.value == other.value
    def __repr__(self) -> str:
        return str(self.value)


class SGD:
    def __init__(self, lr: Real, params: List[Tensor]):
        self.params = params
        self.lr = lr
    def step(self):
        for p in self.params:
            p.value = p.value - self.lr * p.grad
