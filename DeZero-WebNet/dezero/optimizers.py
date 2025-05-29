class Optimizer:
    def __init__(self):
        self.target = None

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        for param in self.target.params():
            if param.grad is not None:
                self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad