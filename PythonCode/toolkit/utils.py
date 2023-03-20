import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import psi


def pseudoinverse(L):
    """
    Input:
    L: the graph Laplacian

    Output:
    PinvL: the Moore-Penrose pseudoinverse of L
    """
    PinvL = np.linalg.inv(L + np.ones(L.shape) / L.shape[0]) - np.ones(L.shape) / L.shape[0]  
    return PinvL


def generate_random_variable(W, take_pseudoinverse=False, graph_type='undirected', normalize=False):
    """
    Input: 
    W: the weighted adjacent matrix
    take_psuedoinverse: whether to take the pseudoinverse of the graph Laplacian
    type: the type of the graph Laplacian, 'undirected' or 'directed_in' or 'directed_out' or 'directed_symmetric'

    Output:
    L: the graph Laplacian
    PinvL: the Moore-Penrose pseudoinverse of L 
    Sigma: the covariance matrix of Gaussian variable
    """
    if graph_type == 'directed_in':
        W = W.T
    elif graph_type == 'directed_symmetric':
        W = W + W.T
    d = np.sum(W, axis=1)
    if normalize:
        d_zeros = np.where(d == 0)
        d[d_zeros] = 1.0
        P = W / d
        P[d_zeros, :] = 1.0
        L = np.eye(W.shape[0]) - P
    else:
        D = np.diag(d)
        L = D - W
    PinvL = pseudoinverse(L)
    if take_pseudoinverse:
        Sigma = PinvL + np.ones(W.shape) / W.shape[0]
    else:
        Sigma = L + np.ones(W.shape) / W.shape[0]
    return L, PinvL, Sigma


def multivar_gaussian_rand_num_generator(mu, sigma, n):
    """
    y = mvg(mu,sigma,n), where mu is mx1 and Sigma is mxm and SPD, produces an mxN matrix y 
    whose columns are samples from the multivariate Gaussian distribution parameterized by 
    mean mu and covariance sigma.

    [y,R] = mvg(mu,sigma,n) also returns the Cholesky factor of the covariance matrix sigma 
    such that sigma = R'*R.

    Input:
    mu: the mean vector, m x 1
    sigma: the covariance matrix, m x m
    n: the number of samples

    Output:
    y: the generated samples, m x n
    R: the Cholesky factor of the covariance matrix sigma
    """
    mu = mu.reshape(-1, 1)
    assert mu.shape[0] == sigma.shape[0], 'The mean vector and the covariance matrix do not have the same dimension'
    assert sigma.shape[0] == sigma.shape[1], 'The covariance matrix is not square'
    assert np.linalg.norm(sigma - sigma.T) < 1e-8, 'The covariance matrix is not symmetric, norm: {}'.format(np.linalg.norm(sigma - sigma.T))

    if np.min(np.linalg.eigvals(sigma)) < 0:
        sigma = sigma + np.eye(sigma.shape[0]) * 1e-8
    R = np.linalg.cholesky(sigma)
    m = mu.shape[0]
    y = np.matmul(R.T, np.random.randn(m, n)) + np.repeat(mu, n, axis=1)
    return y, R


def entropy_estimation(joint_samples, k):
    """
    Input:
    sample_num: the number of samples in random sample generation
    k: the number of nearest neighbors in KNN-based entropy estimation

    Output:
    h: the entropy of the Gaussian variable
    """

    nbrs = NearestNeighbors(n_neighbors=k, metric='chebyshev').fit(joint_samples)
    distances, _ = nbrs.kneighbors(joint_samples)
    r = np.max(distances, axis=1)
    h = psi(joint_samples.shape[0]) + psi(k) + joint_samples.shape[1] * np.mean(np.log(r))
    return h


def convert_to_symmetric_with_zero_diagonal(mat):
    return np.triu(mat, 1) + np.triu(mat, 1).T
