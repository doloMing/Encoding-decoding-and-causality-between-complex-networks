function [HA,HB,HAB,IAB]=MutualInfo(SigmaA,SigmaB,SampleNum,K)
% Input: 
% SigmaA: the covariance matrix Sigma of network A
% SigmaB: the covariance matrix Sigma of network B
% SampleNum: the number of samples in random sample generation
% K: the K value of KNN-based entropy estimation

% Output:
% HA: the entropy of network A
% HB: the entropy of network B
% HAB: the joint entropy of (A,B)
% IAB: the mutual information between network A and network B

[SampleA] = mvg(zeros(size(SigmaA,1),1),SigmaA,SampleNum); % Generate k=SampleNum random samples 
[SampleB] = mvg(zeros(size(SigmaB,1),1),SigmaB,SampleNum); % Generate k=SampleNum random samples
JointSamples=[SampleA',SampleB'];

HA=size(SigmaA,1)/2+size(SigmaA,1)/2*log(2*pi)+1/2*sum(log(eig(SigmaA)));
HB=size(SigmaB,1)/2+size(SigmaB,1)/2*log(2*pi)+1/2*sum(log(eig(SigmaB)));
HAB=EntropyEstimation(JointSamples,K);

IAB=(HA+HB-HAB)*(HA+HB>=HAB);
