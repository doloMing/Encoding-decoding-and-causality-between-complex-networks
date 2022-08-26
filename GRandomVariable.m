function [L,PinvL,Sigma]=GRandomVariable(W,Type)
% Input: 
% W: the weighted adjacent matrix
% Type: the type of covariance matrix (type 1 is L+1/n J, type 2 is
% PinvL+1/n J). Type should be 1 or 2.

% Output:
% L: the graph Laplacian
% PinvL: the Moore–Penrose pseudoinverse of L 
% Sigma: the covariance matrix of Gaussian variable


Deg = sum(W);
L= diag(Deg)-W;
PinvL=PinvFunction(L);

if Type==1
   Sigma=L+ones(size(L,1),size(L,1))./size(L,1);
elseif Type==2
   Sigma=PinvL+ones(size(L,1),size(L,1))./size(L,1); 
end



