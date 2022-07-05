function PinvL=PinvFunction(L)
% Input: 
% L£º the Laplacian of network
% Output: 
% PinvL: the Moore¨CPenrose pseudoinverse of L
PinvL=inv(L+ones(size(L,1),size(L,1)).*1/size(L,1))-ones(size(L,1),size(L,1)).*1/size(L,1);