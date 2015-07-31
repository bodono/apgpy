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

    @property
    def data(self):
        return self

    __rmul__ = __mul__

class NumpyWrapper(IWrapper):
    def __init__(self, nparray):
        self._nparray = nparray

    def dot(self, other):
        return np.inner(self.data, other.data)

    def __add__(self, other):
        return NumpyWrapper(self.data + other.data)

    def __sub__(self, other):
        return NumpyWrapper(self.data - other.data)

    def __mul__(self, scalar):
        return NumpyWrapper(self.data * scalar)

    def copy(self):
        return NumpyWrapper(np.copy(self.data))

    def norm(self):
        return np.linalg.norm(self.data)

    @property
    def data(self):
        return self._nparray

    __rmul__ = __mul__