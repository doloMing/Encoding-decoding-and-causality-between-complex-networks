import numpy as np


def ErdosRenyiNetwork(n, p, seed=None):
    if seed:
        np.random.seed(seed)
    G = np.random.rand(n, n) < p
    G = np.triu(G, 1)
    G = G + G.T
    return G
