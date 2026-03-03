import numpy as np

class Interleaver:
    """
    比特交织器（可逆）
    """

    def __init__(self, size: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.perm = rng.permutation(size)
        self.inv_perm = np.argsort(self.perm)

    def permute(self, bits):
        return bits[:, self.perm]

    def inverse(self, bits):
        return bits[:, self.inv_perm]
