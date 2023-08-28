import numpy as np

from .utils import multivar_gaussian_rand_num_generator, entropy_estimation


def mutual_infomation(sigma_a, sigma_b, sample_num, k):
    """
    Input:
    sigma_a: the covariance matrix Sigma of network a
    sigma_b: the covariance matrix Sigma of network b
    sample_num: the number of samples in random sample generation
    k: the number of nearest neighbors in KNN-based entropy estimation

    Output:
    mi: the mutual information between a and b
    """

    sample_a = multivar_gaussian_rand_num_generator(np.zeros(sigma_a.shape[0]), sigma_a, sample_num)[0]
    sample_b = multivar_gaussian_rand_num_generator(np.zeros(sigma_b.shape[0]), sigma_b, sample_num)[0]
    joint_samples = np.concatenate((sample_a, sample_b), axis=0).T
    
    h_a = (1.0 + np.log(2 * np.pi)) * sigma_a.shape[0] / 2.0 + np.sum(np.log(np.linalg.eigvals(sigma_a))) / 2.0
    h_b = (1.0 + np.log(2 * np.pi)) * sigma_b.shape[0] / 2.0 + np.sum(np.log(np.linalg.eigvals(sigma_b))) / 2.0
    h_ab = entropy_estimation(joint_samples, k)

    h_a = np.real((1.0 + np.log(2 * np.pi)) * sigma_a.shape[0] / 2.0 + np.sum(np.log(np.linalg.eigvals(sigma_a))) / 2.0)
    h_b = np.real((1.0 + np.log(2 * np.pi)) * sigma_b.shape[0] / 2.0 + np.sum(np.log(np.linalg.eigvals(sigma_b))) / 2.0)

    if h_a<0.0:
        h_a=0.0

    if h_b<0.0:
        h_b=0.0

    h_ab = entropy_estimation(joint_samples, k)

    mi = np.min([np.max([h_a + h_b - h_ab, 0]),h_a,h_b])

    return h_a, h_b, h_ab, mi
