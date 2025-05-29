import numpy as np

class Variable:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx
                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None

    def __add__(self, other):
        other = as_variable(other)
        return Add()(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = as_variable(other)
        return Sub()(self, other)

    def __rsub__(self, other):
        other = as_variable(other)
        return Sub()(other, self)

    def __mul__(self, other):
        other = as_variable(other)
        return Mul()(self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = as_variable(other)
        return Div()(self, other)

    def __rtruediv__(self, other):
        other = as_variable(other)
        return Div()(other, self)

    def __neg__(self):
        return Neg()(self)

    def sum(self):
        return Sum()(self)

    def reshape(self, *shape):
        return Reshape(shape)(self)

    def dot(self, other):
        return Dot()(self, other)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Model:
    def __init__(self):
        self._params = []

    def __setattr__(self, name, value):
        if isinstance(value, Variable):
            self._params.append(value)
        super().__setattr__(name, value)

    def params(self):
        return self._params

    def cleargrads(self):
        for p in self._params:
            p.cleargrad()

    def plot(self, x, to_file='model.dot'):
        with open(to_file, 'w') as f:
            f.write('// dummy graph\n')

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    def backward(self, *gys):
        gy = gys[0]
        return gy, gy

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    def backward(self, *gys):
        gy = gys[0]
        return gy, -gy

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    def backward(self, *gys):
        gy = gys[0]
        x0, x1 = self.inputs
        return gy * x1.data, gy * x0.data

class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1
    def backward(self, *gys):
        gy = gys[0]
        x0, x1 = self.inputs
        return gy / x1.data, gy * (-x0.data / (x1.data ** 2))

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, *gys):
        gy = gys[0]
        return (-gy,)

class Sum(Function):
    def forward(self, x):
        self.x_shape = x.shape
        return x.sum()

    def backward(self, *gys):
        gy = gys[0]
        return gy * np.ones(self.x_shape, dtype=gy.dtype)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, *gys):
        return gys[0].reshape(self.input_shape)

class Dot(Function):
    def forward(self, x, W):
        self.x, self.W = x, W
        return x.dot(W)

    def backward(self, *gys):
        gx = gys[0].dot(self.W.T)
        gW = self.x.T.dot(gys[0])
        return gx, gW

__all__ = ["Variable", "Function", "Model"]