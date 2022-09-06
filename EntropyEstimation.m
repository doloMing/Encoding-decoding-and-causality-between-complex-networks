function H=EntropyEstimation(JointSamples,K)
% Input: 
% JointSamples: the number of generated random samples, where columns
% correspond to variables and rows correspond to observations
% K: the K value of KNN-based entropy estimation

% Output:
% H: the estimated entropy

[~,D]=knnsearch(JointSamples,JointSamples,'Distance','chebychev','K',K,'IncludeTies',true);
M=cell2mat(D);
R=max(M,[],2);
H=psi(size(JointSamples,1))-psi(K)+log(1)+size(JointSamples,2)*mean(log(R));

