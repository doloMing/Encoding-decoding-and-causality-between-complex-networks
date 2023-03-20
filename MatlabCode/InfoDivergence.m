function [DAB,DBA]=InfoDivergence(NSigmaA,NSigmaB)
% Input: 
% NSigmaA: the covariance matrix Sigma of network A
% NSigmaB: the covariance matrix Sigma of network B

% Output:
% DAB: the information divergence from network A to network B
% DBA: the information divergence from network B to network A
% Note that DAB may be different from DBA. Information divergence is 
% directional

%% Calculate DAB 
DAB=real(0.5*(trace(NSigmaA/NSigmaB)-size(NSigmaA,1)+log(prod(eig(NSigmaB)./eig(NSigmaA)))));
%% Calculate DBA
DBA=real(0.5*(trace(NSigmaB/NSigmaA)-size(NSigmaB,1)+log(prod(eig(NSigmaA)./eig(NSigmaB)))));
