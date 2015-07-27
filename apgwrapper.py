import numpy as np

class IWrapper:
    def dot(self, other):
        raise NotImplementedError("Implement in subclass")

    def __add__(self, other):
        raise NotImplementedError("Implement in subclass")

    def __sub__(self, other):
        raise NotImplementedError("Implement in subclass")

    def __mul__(self, scalar):
        raise NotImplementedError("Implement in subclass")

    def copy(self):
        raise NotImplementedError("Implement in subclass")

    def norm(self):
        raise NotImplementedError("Implement in subclass")

    __rmul__ = __mul__

class NumpyWrapper(IWrapper):
    def __init__(self, nparray):
        self.nparray = nparray

    def dot(self, other):
        return np.inner(self.nparray, other.nparray)

    def __add__(self, other):
        return NumpyWrapper(self.nparray + other.nparray)

    def __sub__(self, other):
        return NumpyWrapper(self.nparray - other.nparray)

    def __mul__(self, scalar):
        return NumpyWrapper(self.nparray * scalar)

    def copy(self):
        return NumpyWrapper(np.copy(self.nparray))

    def norm(self):
        return np.linalg.norm(self.nparray)

    __rmul__ = __mul__