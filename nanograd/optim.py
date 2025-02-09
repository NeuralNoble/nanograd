from .engine import Value

class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0

class GD(Optimizer):  # Changed name from BGD to GD (just gradient descent)
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

class SGD(Optimizer):
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad  # True SGD is controlled in the training loop
