function [L,PinvL,Sigma]=GMRandomField(W)
% Input: 
% W: the weighted adjacent matrix

% Output:
% L: the graph Laplacian
% PinvL: the Moore¨CPenrose pseudoinverse of L 
% Sigma: the covariance matrix of Gaussian Markov random field

Deg = sum(W);
L= diag(Deg)-W;
PinvL=PinvFunction(L);
Sigma=L+ones(size(L,1),size(L,1))./size(L,1);

