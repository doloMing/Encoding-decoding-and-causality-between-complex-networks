import numpy as np
from scipy.io import savemat

from random_networks import ErdosRenyiNetwork, WattsStrogatzNetwork
from toolkit import infomation_divergence, mutual_infomation, fisher_information, \
    granger_causality_and_transfer_entropy, generate_random_variable, \
    network_approximation, convert_to_symmetric_with_zero_diagonal


def run(config):
    network_a = WattsStrogatzNetwork(300, 15, 0.5)
    weight_base_a = 1 + np.random.rand(network_a.shape[0], network_a.shape[1]) * 9
    W_a = network_a * (np.triu(weight_base_a, 1) + np.triu(weight_base_a, 1).T)
    # np.save('W_a.npy', W_a)

    network_b = ErdosRenyiNetwork(500, 0.2, seed=100)
    weight_base_b = 1 + np.random.rand(network_b.shape[0], network_b.shape[1]) * 9
    W_b = network_b * (np.triu(weight_base_b, 1) + np.triu(weight_base_b, 1).T)
    # np.save('W_b.npy', W_b)

    # W_a = np.load('W_a.npy')
    # W_b = np.load('W_b.npy')
    savemat('data.mat', {'W_a': W_a, 'W_b': W_b})

    L_a, PinvL_a, Sigma_a = generate_random_variable(W_a, take_pseudoinverse=False)
    L_b, PinvL_b, Sigma_b = generate_random_variable(W_b, take_pseudoinverse=False)
    # print(Sigma_a)
    # print(Sigma_b)

    net_approx = network_approximation(W_a, W_b, L_a, L_b, PinvL_a, PinvL_b, Sigma_a, Sigma_b)
    new_sigma_a = net_approx['new_sigma_a']
    new_sigma_b = net_approx['new_sigma_b']

    # information divergence
    divergence_ab, divergence_ba = infomation_divergence(new_sigma_a, new_sigma_b)
    # print(new_sigma_a)
    # print(np.linalg.det(new_sigma_a))
    print('divergence_ab: {}'.format(divergence_ab))
    print('divergence_ba: {}'.format(divergence_ba))
    print()

    # mutual information
    sample_num = 5000
    k = 2
    H_a, H_b, H_ab, I_ab = mutual_infomation(Sigma_a, Sigma_b, sample_num, k)
    print('H_a: {}'.format(H_a))
    print('H_b: {}'.format(H_b))
    print('H_ab: {}'.format(H_ab))
    print('I_ab: {}'.format(I_ab))
    print()

    # fisher information
    deg_new_network = np.sum(W_b, axis=0)
    theta_number = 10
    o_number = 100
    theta_mat = np.zeros((o_number, theta_number))
    for i in range(o_number):
        random_id_b = np.random.permutation(W_b.shape[0])
        theta_mat[i] = deg_new_network[random_id_b[:theta_number]]
    theta_mat = np.unique(theta_mat, axis=0)

    sigma_ensemble = np.zeros((o_number, Sigma_a.shape[0], Sigma_a.shape[1]))
    for i in range(o_number):
        weight_base = np.abs(np.random.normal(0, np.mean(theta_mat[i], axis=0), Sigma_a.shape))
        _, _, noise_sigma_a1 = generate_random_variable(W_a * convert_to_symmetric_with_zero_diagonal(weight_base))
        sigma_ensemble[i] = noise_sigma_a1
    
    # print(sigma_ensemble.shape)
    fisher_info_mat = fisher_information(sigma_ensemble, theta_mat)
    print('fisher_info_mat: {}'.format(fisher_info_mat))
    print()

    # granger causality and transfer entropy
    rand_p_num = 20
    g_ab_vec, g_ab, t_ab_vec, t_ab, size_ab_vec = granger_causality_and_transfer_entropy(Sigma_a, Sigma_b, 
                                                    sample_num, rand_p_num, k)
    g_ba_vec, g_ba, t_ba_vec, t_ba, size_ba_vec = granger_causality_and_transfer_entropy(Sigma_b, Sigma_a,
                                                    sample_num, rand_p_num, k)
    # print('t_ab_vec: {}'.format(t_ab_vec))
    print('t_ab: {}'.format(t_ab))
    # print('g_ab_vec: {}'.format(g_ab_vec))
    print('g_ab: {}'.format(g_ab))
    # print('size_ab_vec: {}'.format(size_ab_vec))
    print()
    # print('t_ba_vec: {}'.format(t_ba_vec))
    print('t_ba: {}'.format(t_ba))
    # print('g_ba_vec: {}'.format(g_ba_vec))
    print('g_ba: {}'.format(g_ba))
    # print('size_ba_vec: {}'.format(size_ba_vec))
