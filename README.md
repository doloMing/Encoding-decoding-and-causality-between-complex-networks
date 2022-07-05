# Analytic-relations-between-complex-networks-encoding-decoding-and-causality
This is the toolbox introduced in the paper entitled as "Analytic relations between complex networks: encoding, decoding, and causality".

In general, this toolbox can be used to deal with following questions:
A. Represent a complex network by a Gaussian Markov random field, where the infomation of network topology is contained in the covariance matrix of the Gaussian Markov random field
B. Analytically calculate the similarity between complex networks from the perspectives of encoding, decoding, and causal analyses
C. Help solve network clustering and classification

In the released files, you can find key functions used in our research:
1. GMRandomField.m: The function used to work out the graph Laplacian (the discrete Schr\"{o}dinger operator), the Moore–Penrose pseudoinverse of L, the covariance matrix of Gaussian Markov random field
2. NetworkApproximation.m: The function used to realize network approximation if two networks have different size
3. InfoDivergence.m: The function used to work out the information divergence between two networks
4. MutualInfo.m: The function used to work out the mutual information between two networks
5. FisherInfo.m: The function used to work out the Fisher information between two networks
6. GrangerCandTransferE.m: The function used to work out the Granger causality and transfer entropy between two networks

Apart from these functions, we also provide three random network models used in our research. Please note that these models are developed by previous studies cited in our paper.
7. BAModel.m: The function used to generate a Barab\'{a}si–Albert network
8. ErdosRenyi.m: The function used to generate a Erdos-Renyi network
9. WattsStrogatz.m: The function used to generate a Watts-Strogatz network

To help users understand how to use our functions, we provide a simple instance in the released files:
10. Instance.m: The function used to run the simple instance

