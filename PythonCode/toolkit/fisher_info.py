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
    n = sigma_ensemble.shape[1]
    
    derivatives = []
    for t in range(k):
        derivative_matrix = np.zeros((x-1, n, n))
        for i in range(n):
            for j in range(n):
                needed_s = sigma_ensemble[:, i, j]
                # print(needed_s)
                derivative_vec = np.diff(needed_s, axis=0) / (np.diff(theta_matrix[:, t], axis=0) + np.finfo(float).eps)
                # print(derivative_vec)
                derivative_vec[np.diff(theta_matrix[:, t], axis=0) == 0] = 0
                derivative_matrix[:, i, j] = derivative_vec
        derivatives.append(derivative_matrix)
    derivatives = np.array(derivatives)

    fisher_info_matrix = np.zeros((x-1, k, k))
    for i in range(x-1):
        for j in range(k):
            derivative_matrix_1 = derivatives[j]
            for l in range(k):
                derivative_matrix_2 = derivatives[l]
                sigma_ensemble_i_inv = np.linalg.inv(sigma_ensemble[i])
                fisher_info_matrix[i, j, l] = 0.5 * np.trace(np.matmul(sigma_ensemble_i_inv, 
                        np.matmul(derivative_matrix_1[i], np.dot(sigma_ensemble_i_inv, derivative_matrix_2[i]))))
                
    return fisher_info_matrix