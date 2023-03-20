import numpy as np


def infomation_divergence(sigma_a, sigma_b):
    """
    Input:
    sigma_a: the covariance matrix Sigma of network a
    sigma_b: the covariance matrix Sigma of network b

    Output:
    d_ab: the information divergence from a to b
    d_ba: the information divergence from b to a
    """

    # d_ab = 0.5*(np.trace(np.matmul(np.linalg.inv(sigma_b), sigma_a)) - sigma_a.shape[0] 
    #             + np.log(np.linalg.det(sigma_b)/np.linalg.det(sigma_a)))
    d_ab = 0.5*(np.trace(np.matmul(np.linalg.inv(sigma_b), sigma_a)) - sigma_a.shape[0] 
                + np.log(np.prod(np.linalg.eigvals(sigma_b)/np.linalg.eigvals(sigma_a))))
    # d_ba = 0.5*(np.trace(np.matmul(np.linalg.inv(sigma_a), sigma_b)) - sigma_b.shape[0] 
    #             + np.log(np.linalg.det(sigma_a)/np.linalg.det(sigma_b)))
    d_ba = 0.5*(np.trace(np.matmul(np.linalg.inv(sigma_a), sigma_b)) - sigma_b.shape[0] 
                + np.log(np.prod(np.linalg.eigvals(sigma_a)/np.linalg.eigvals(sigma_b))))
    return d_ab, d_ba
