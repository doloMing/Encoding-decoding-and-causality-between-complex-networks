import numpy as np


def fisher_information(sigma_ensemble, theta_matrix):
    """
    Input:
    sigma_ensemble: x*n*n matrix, each n*n matrix is a covariance matrix Sigma 
        of the Gaussian Markov random field, corresponding to a covariance matrix 
        Sigma controlled by the observation of Theta.
    theta_matrix: x*k, where each row contains an observation of Theta, a 1*k vector 
        Theta=(theta_1,...,theta_k). Please note that this function expects a pre-processed 
        ThetaMatrix, where each observarion of Theta is unique and all observations are 
        sorted according to an appropriate criterion. Otherwise, there is no information 
        used to define partial derivatives in the equation of Fisher information. In an 
        sorted matrix, the partial derivatives are calculated based on every pair of adjacent 
        rows in ThetaMatrix.

    Output:
    fisher_info: (x-1)*k*k matrix of Fisher information
    """

    assert theta_matrix.shape[0] == sigma_ensemble.shape[0]
    x, k = theta_matrix.shape
    
    derivatives = []
    sigma_ensemble_diff = np.diff(sigma_ensemble, axis=0)
    for t in range(k):
        derivative_matrix = sigma_ensemble_diff / (np.diff(theta_matrix[:, t], axis=0) + np.finfo(float).eps).reshape(-1,1,1)
        derivative_matrix[np.diff(theta_matrix[:, t], axis=0) == 0] = 0
        derivatives.append(derivative_matrix)
    derivatives = np.array(derivatives)

    fisher_info_matrix = np.zeros((x-1, k, k))
    for i in range(x-1):
        sigma_ensemble_i_inv = np.linalg.inv(sigma_ensemble[i])
        for j in range(k):
            derivative_matrix_1 = derivatives[j, i]
            for l in range(k):
                derivative_matrix_2 = derivatives[l, i]
                fisher_info_matrix[i, j, l] = 0.5 * np.trace(np.matmul(sigma_ensemble_i_inv, 
                        np.matmul(derivative_matrix_1, np.dot(sigma_ensemble_i_inv, derivative_matrix_2))))
    return fisher_info_matrix
