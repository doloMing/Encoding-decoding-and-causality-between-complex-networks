function [FIMatrix]=FisherInfo(SigmaEnsemble,ThetaMatrix)
% Input: 
% (1) SigmaEnsemble should be a x*n*n matrix, where each n*n matrix is a
% covariance matrix Sigma of the Gaussian Markov random field. 
% Each n*n matrix in SigmaEnsemble corresponds to a covariance matrix 
% Sigma controlled by the observation of Theta.

% (2) ThetaMatrix should be a x*k matrix, where each row contains an
% observation of Theta, a 1*k vector Theta=(theta_1,...,theta_k). Please
% note that this function expects a pre-processed ThetaMatrix, where each
% observarion of Theta is unique and all observations are sorted according
% to an appropriate criterion. Otherwise, there is no information used to
% define partial derivatives in the equation of Fisher information. In an
% sorted matrix, the partial derivatives are calculated based on every pair
% of adjacent rows in ThetaMatrix.

% Output:
% (1) FIMatrix is a (x-1)k*k matrix of Fisher information

% Step A: Generate derivative of Sigma with respect to each theta_i
DerivativeCell=cell(size(ThetaMatrix,2),1);
for IDT=1:size(ThetaMatrix,2)
    ThetaI=ThetaMatrix(:,IDT);
    DerivativeMatrix=zeros(size(ThetaMatrix,1)-1,size(SigmaEnsemble,2),size(SigmaEnsemble,2));
    for ID1=1:size(SigmaEnsemble,2)
        for ID2=1:size(SigmaEnsemble,2)
            NeededS=squeeze(SigmaEnsemble(:,ID1,ID2));
            DerivativeVec=diff(NeededS)./(diff(ThetaI)+eps);
            DerivativeVec(diff(ThetaI)==0)=0;
            DerivativeMatrix(:,ID1,ID2)=DerivativeVec; 
        end
    end
    DerivativeCell{IDT,1}=DerivativeMatrix;
end

% Step B: Generate Fisher information matrix
FIMatrix=zeros(size(ThetaMatrix,1)-1,size(ThetaMatrix,2),size(ThetaMatrix,2));
for ID0=1:size(ThetaMatrix,1)-1
    for ID1=1:size(ThetaMatrix,2)
        DerivativeMatrix1=DerivativeCell{ID1,1};
        for ID2=1:size(ThetaMatrix,2)
            DerivativeMatrix2=DerivativeCell{ID2,1};
            FIMatrix(ID0,ID1,ID2)=1/2*trace(squeeze(SigmaEnsemble(ID0,:,:))\squeeze(DerivativeMatrix1(ID0,:,:))/squeeze(SigmaEnsemble(ID0,:,:))*squeeze(DerivativeMatrix2(ID0,:,:)));
        end
    end
end

