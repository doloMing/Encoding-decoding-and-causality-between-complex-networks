import random
import numpy as np

from .utils import multivar_gaussian_rand_num_generator, entropy_estimation


def granger_causality_and_transfer_entropy(sigma_a, sigma_b, sample_num, rand_partition_num, k):
    """
    compute Granger causality and transfer entropy from network A to network B

    Input:
    sigma_a: the covariance matrix Sigma of network a
    sigma_b: the covariance matrix Sigma of network b
    sample_num: the number of samples in random sample generation
    random_p_num: the number of repetitions of random partition
    k: the number of nearest neighbors in KNN-based entropy estimation

    Output:
    granger_causality_ab_vec: the vector of Granger causality values from network A to 
        network B corresponding to different random partitions on network B
    granger_causality_ab: the averaged Granger causality value from network A to network B
    transfer_entropy_ab_vec: the vector of transfer entropy values from network A to
        network B corresponding to different random partitions on network B
    transfer_entropy_ab: the averaged transfer entropy value from network A to network B
    """

    size_ab_vec = np.zeros(rand_partition_num, dtype=int)
    granger_causality_ab_vec = np.zeros(rand_partition_num)
    transfer_entropy_ab_vec = np.zeros(rand_partition_num)
    for i in range(rand_partition_num):
        random_node = np.random.permutation(sigma_b.shape[0])
        size_ab_vec[i] = random.randint(1, sigma_b.shape[0]-1)
        subnet_b1 = sigma_b[random_node[:size_ab_vec[i]]][:, random_node[:size_ab_vec[i]]]
        subnet_b2 = sigma_b[random_node[size_ab_vec[i]:]][:, random_node[size_ab_vec[i]:]]

        # transfer entropy
        detnet_b = np.sum(np.log(np.linalg.eigvals(sigma_b)))
        h_b = sigma_b.shape[0] * 0.5 * (1 + np.log(2*np.pi)) + 0.5 * detnet_b
        
        sample_a, _ = multivar_gaussian_rand_num_generator(np.zeros(sigma_a.shape[0]), sigma_a, sample_num)
        sample_b1, _ = multivar_gaussian_rand_num_generator(np.zeros(subnet_b1.shape[0]), subnet_b1, sample_num)
        joint_samples = np.concatenate((sample_a, sample_b1), axis=0).T
        h_a_sb1 = entropy_estimation(joint_samples, k)

        h_sb1 = entropy_estimation(sample_b1.T, k)

        sample_b, _ = multivar_gaussian_rand_num_generator(np.zeros(sigma_b.shape[0]), sigma_b, sample_num)
        joint_samples = np.concatenate((sample_a, sample_b), axis=0).T
        h_ab = entropy_estimation(joint_samples, k)

        transfer_entropy_ab_vec[i] = h_b + h_a_sb1 - h_sb1 - h_ab

        # granger causality
        sigma_1 = subnet_b1 - np.matmul(np.matmul(sigma_b[random_node[:size_ab_vec[i]]][:, random_node[size_ab_vec[i]:]],
            np.linalg.inv(subnet_b2)) , sigma_b[random_node[size_ab_vec[i]:]][:, random_node[:size_ab_vec[i]]])
        samples_b1_a = np.concatenate((sample_b1, sample_a), axis=0).T
        samples_b2, _ = multivar_gaussian_rand_num_generator(np.zeros(subnet_b2.shape[0]), subnet_b2, sample_num)
        samples_b2 = samples_b2.T
        cov_b2_b1_a = np.zeros((samples_b2.shape[1], samples_b1_a.shape[1]))
        for j in range(samples_b2.shape[1]):
            for k in range(samples_b1_a.shape[1]):
                cov_b2_b1_a[j, k] = np.cov(samples_b2[:, j], samples_b1_a[:, k])[0, 1]
        cov_b1_a = np.zeros((samples_b1_a.shape[1], samples_b1_a.shape[1]))
        for j in range(samples_b1_a.shape[1]):
            for k in range(samples_b1_a.shape[1]):
                cov_b1_a[j, k] = np.cov(samples_b1_a[:, j], samples_b1_a[:, k])[0, 1]
        sigma_2 = subnet_b2 - np.dot(np.dot(cov_b2_b1_a, np.linalg.inv(cov_b1_a)), cov_b2_b1_a.T)
        # print(sigma_1)
        # print(np.linalg.eigvals(sigma_1))
        # print(np.linalg.eigvals(sigma_2))
        # print()
        eigvals_sigma_1 = np.linalg.eigvals(sigma_1)
        eigvals_sigma_1[eigvals_sigma_1 <= 0] = 1
        eigvals_sigma_2 = np.linalg.eigvals(sigma_2)
        eigvals_sigma_2[eigvals_sigma_2 <= 0] = 1
        granger_causality_ab_vec[i] = np.sum(np.log(eigvals_sigma_1)) - np.sum(np.log(eigvals_sigma_2))

    transfer_entropy_ab = np.mean(transfer_entropy_ab_vec)
    granger_causality_ab = np.mean(granger_causality_ab_vec)

    return granger_causality_ab_vec, granger_causality_ab, transfer_entropy_ab_vec, transfer_entropy_ab, size_ab_vec
