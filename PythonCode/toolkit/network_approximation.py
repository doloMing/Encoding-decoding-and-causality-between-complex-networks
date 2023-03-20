import numpy as np

from .utils import pseudoinverse


def network_approximation(W_a, W_b, L_a, L_b, PinvL_a, PinvL_b, sigma_a, sigma_b):
    """
    This function implements network approximation if the two networks have
    different sizes (contain different numbers of nodes). 
    The approximation is defined based on Laplacian energy. 

    Input: 
    WA: the weighted adjacent matrix of network A
    WB: the weighted adjacent matrix of network B
    LA: the Laplacian of network A
    LB: the Laplacian of network B
    PinvLA: the Moore–Penrose pseudoinverse of LA
    PinvLB: the Moore–Penrose pseudoinverse of LB
    SigmaA: the covariance matrix Sigma of network A
    SigmaB: the covariance matrix Sigma of network B

    Output:
    NLA: new Laplacian matrix of network A. If network A has a larger size,
    then NLA \neq LA because it will be modified by approximation. Otherwise,
    NLA=LA.
    NLB: new Laplacian matrix of network B. If network B has a larger size,
    then NLB \neq LB because it will be modified by approximation. Otherwise,
    NLB=LB.
    NPinvLA: the Moore–Penrose pseudoinverse of NLA
    NPinvLB: the Moore–Penrose pseudoinverse of NLB
    NSigmaA: the new covariance matrix of Gaussian Markov random field of
    network A
    NSigmaB: the new covariance matrix of Gaussian Markov random field of
    network B
    Gamma: the rationality of approximation
    """

    if W_a.shape[0] == W_b.shape[0]:
        new_L_a = L_a
        new_L_b = L_b
        new_PinvL_a = PinvL_a
        new_PinvL_b = PinvL_b
        new_sigma_a = sigma_a
        new_sigma_b = sigma_b
        sort_LE_based_c = np.zeros(W_a.shape[0])
        index = np.zeros(W_a.shape[0])
        LE = 0
        new_LE = 0
        gamma = 1
    elif W_a.shape[0] > W_b.shape[0]:
        new_L_a, new_PinvL_a, new_sigma_a, sort_LE_based_c, index, LE, new_LE, gamma \
            = approximate(W_a, W_b.shape[0])
        new_L_b = L_b
        new_PinvL_b = PinvL_b
        new_sigma_b = sigma_b
    else:
        new_L_b, new_PinvL_b, new_sigma_b, sort_LE_based_c, index, LE, new_LE, gamma \
            = approximate(W_b, W_a.shape[0])
        new_L_a = L_a
        new_PinvL_a = PinvL_a
        new_sigma_a = sigma_a

    results = {
        'new_L_a': new_L_a,
        'new_L_b': new_L_b,
        'new_PinvL_a': new_PinvL_a,
        'new_PinvL_b': new_PinvL_b,
        'new_sigma_a': new_sigma_a,
        'new_sigma_b': new_sigma_b,
        'sort_LE_based_c': sort_LE_based_c,
        'index': index,
        'LE': LE,
        'new_LE': new_LE,
        'gamma': gamma
    }
    return results
        

def approximate(W_a, W_b_shape):
    deg_a = np.sum(W_a, axis=0) ** 2
    W_a_with_zero_diag = W_a - np.diag(np.diag(W_a))
    LE = np.sum(deg_a, axis=0) + np.sum(W_a_with_zero_diag ** 2)
    delta_LE = np.zeros(W_a.shape[0])
    W_a_square = np.matmul(W_a, W_a)
    W_a_square_with_zero_diag = W_a_square - np.diag(np.diag(W_a_square))
    for i in range(W_a.shape[0]):
        M_a = np.delete(np.delete(W_a, i, 0), i, 1)
        M_a_square = np.matmul(M_a, M_a)
        M_a_square_with_zero_diag = M_a_square - np.diag(np.diag(M_a_square))
        delta_LE[i] = 4 * W_a_square[i, i] + 2 * np.sum(W_a_square_with_zero_diag) \
                        - np.sum(M_a_square_with_zero_diag)
    LE_based_c = delta_LE / LE
    sort_LE_based_c = np.sort(LE_based_c)
    index = np.argsort(LE_based_c)
    needed_nodes = index[-W_b_shape:]
    W_a = W_a[needed_nodes][:, needed_nodes]
    deg_a = np.sum(W_a, axis=0)
    new_L_a = np.diag(deg_a) - W_a
    new_PinvL_a = pseudoinverse(new_L_a)
    new_sigma_a = new_L_a + np.ones(new_PinvL_a.shape) / new_PinvL_a.shape[0]
    deg_a_sum_square = np.sum(W_a, axis=0) ** 2
    W_a_with_zero_diag = W_a - np.diag(np.diag(W_a))
    new_LE = np.sum(deg_a_sum_square, axis=0) + np.sum(W_a_with_zero_diag ** 2)
    gamma = new_LE / LE
    return new_L_a, new_PinvL_a, new_sigma_a, sort_LE_based_c, index, LE, new_LE, gamma
