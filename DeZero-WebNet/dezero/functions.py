# functions.py
from .core import Function
import numpy as np

class LinearFunction(Function):
    def forward(self, x, W, b):
        self.x = x
        self.W = W
        return x.dot(W) + b

    def backward(self, *gys):
        gy, = gys
        gx = gy.dot(self.W.T)
        gW = self.x.T.dot(gy)
        gb = gy.sum(axis=0, keepdims=True)
        return gx, gW, gb

class ReLU(Function):
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, *gys):
        gy, = gys
        gy[self.mask] = 0
        return gy

def relu(x):
    return ReLU()(x)

def linear(x, W, b):
    return LinearFunction()(x, W, b)

class MeanSquaredError(Function):
    def forward(self, y, t):
        diff = y - t
        self.diff = diff
        return np.mean(diff ** 2)

    def backward(self, *gys):
        gy, = gys
        batch_size = len(self.diff)
        return gy * 2 * self.diff / batch_size, -gy * 2 * self.diff / batch_size

def mean_squared_error(y, t):
    return MeanSquaredError()(y, t)