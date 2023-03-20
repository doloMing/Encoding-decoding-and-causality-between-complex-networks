import numpy as np


def WattsStrogatzNetwork(n, k, beta):
    s = np.repeat(np.arange(1, n+1).reshape(-1, 1), k, 1)
    t = s + np.repeat(np.arange(1, k+1).reshape(1, -1), n, 0)
    t = (t-1) % n + 1

    # for source in range(n):
    #     switch_edge = np.random.rand(k) < beta
        
    #     new_targets = np.random.rand(n)
    #     new_targets[source] = 0
    #     new_targets[s[t==source+1]-1] = 0
    #     new_targets[t[source, ~switch_edge]-1] = 0

    #     ind = np.argsort(new_targets)[::-1]
    #     t[source, switch_edge] = ind[:np.count_nonzero(switch_edge)]

    graph_edges = np.concatenate((s.reshape(-1, 1), t.reshape(-1, 1)), 1)
    adjancency_matrix = np.zeros((n, n))
    adjancency_matrix[graph_edges[:, 0]-1, graph_edges[:, 1]-1] = 1
    adjancency_matrix[graph_edges[:, 1]-1, graph_edges[:, 0]-1] = 1

    return adjancency_matrix
