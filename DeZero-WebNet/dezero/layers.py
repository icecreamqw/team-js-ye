# layers.py
import numpy as np
from .core import Variable
from .models import Layer
from .functions import linear
import dezero.functions as F

class Linear(Layer):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size
        self.W = None
        self.b = None

    def __call__(self, x):
        if self.W is None:
            in_size = x.data.shape[1]
            W_data = np.random.randn(in_size, self.out_size) * 0.01
            self.W = Variable(W_data)
            self.b = Variable(np.zeros(self.out_size))
            self.add_param('W', self.W)
            self.add_param('b', self.b)
        return linear(x, self.W, self.b)

class ReLU:
    def __call__(self, x):
        return F.relu(x)

class Sigmoid:
    def __call__(self, x):
        return F.sigmoid(x)

class Tanh:
    def __call__(self, x):
        return F.tanh(x)

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.train = True  # Default to training mode

    def __call__(self, x):
        return F.dropout(x, self.dropout_ratio, self.train)