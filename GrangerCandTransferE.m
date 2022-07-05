function [TABVec,TAB,GABVec,GAB,SizeABVec]=GrangerCandTransferE(SigmaA,SigmaB,SampleNum,RandomPNum,K)
% Input: 
% SigmaA: the covariance matrix Sigma of network A
% SigmaB: the covariance matrix Sigma of network B
% SampleNum: the number of samples in random sample generation
% RandomPNum: the number of repetitions of random partition
% K: the K value of KNN-based entropy estimation

% Output:
% TABVec: the vector of transfer entropy values from network A to network B corresponding to different
% random partitions on network B
% TAB: the averaged transfer entropy value from network A to network B
% GABVec: the vector of Granger causality values from network A to network B corresponding to different
% random partitions on network B
% GAB: the averaged Granger causality value from network A to network B
% SizeABVec: Size vector of randomly selected sub-network of network B 

SizeABVec=zeros(1,RandomPNum);
TABVec=zeros(1,RandomPNum);
GABVec=zeros(1,RandomPNum);
for ID=1:RandomPNum
    %% Transfer entropy
    % Random partition on network B
    RandomNode=randperm(size(SigmaB,1));
    SizeABVec(ID)=randi([1,size(SigmaB,1)],1,1);
    SubNetB1=SigmaB(RandomNode(1:SizeABVec(ID)),RandomNode(1:SizeABVec(ID)));
    SubNetB2=SigmaB(RandomNode(SizeABVec(ID)+1:end),RandomNode(SizeABVec(ID)+1:end));
    % H(SubNetB2,SubNetB1)
    DetNetB=sum(log(eig(SigmaB)));
    HB=size(SigmaB,1)/2+size(SigmaB,1)/2*log(2*pi)+1/2*DetNetB;
    % H(SubNetB1,Network A)
    [SampleA] = mvg(zeros(size(SigmaA,1),1),SigmaA,SampleNum); % Generate k=SampleNum random samples 
    [SampleB1] = mvg(zeros(size(SubNetB1,1),1),SubNetB1,SampleNum); % Generate k=SampleNum random samples
    JointSamples=[SampleA',SampleB1'];
    HASB1=EntropyEstimation(JointSamples,K);
    % H(SubNetB1)
    HSB1=EntropyEstimation(SampleB1',K);
    % H(AB)
    [SampleB] = mvg(zeros(size(SigmaB,1),1),SigmaB,SampleNum); % Generate k=SampleNum random samples
    JointSamples=[SampleA',SampleB'];
    HAB=EntropyEstimation(JointSamples,K);
    TABVec(ID)=HB+HASB1-HSB1-HAB;
    
    %% Granger causality
    % Covariance matrix of the residuals without Network A
    Sigma1=SubNetB1-SigmaB(RandomNode(1:SizeABVec(ID)),RandomNode(SizeABVec(ID)+1:end))*inv(SubNetB2)*SigmaB(RandomNode(SizeABVec(ID)+1:end),RandomNode(1:SizeABVec(ID)));
    % Covariance matrix of the residuals with Network A
    % To calculate this matrix, we need to follow Eq.(1) in "Granger
    % Causality and Transfer Entropy Are Equivalent for Gaussian Variables"
    % Covariance between SubNetB2 and (SubNetB1,Network A)
    SamplesB1A=[SampleB1',SampleA'];
    [SamplesB2]=mvg(zeros(size(SubNetB2,1),1),SubNetB2,SampleNum);
    SamplesB2=SamplesB2';
    CovB2B1A=zeros(size(SamplesB2,2),size(SamplesB1A,2));
    for ID1=1:size(SamplesB2,2)
        for ID2=1:size(SamplesB1A,2)
            CovM=cov(SamplesB2(:,ID1),SamplesB1A(:,ID2));
            CovB2B1A(ID1,ID2)=CovM(1,2);
        end
    end
    CovB1A=zeros(size(SamplesB1A,2),size(SamplesB1A,2));
    for ID1=1:size(SamplesB1A,2)
        for ID2=1:size(SamplesB1A,2)
            CovM=cov(SamplesB1A(:,ID1),SamplesB1A(:,ID2));
            CovB1A(ID1,ID2)=CovM(1,2);
        end
    end
    Sigma2=SubNetB2-CovB2B1A*inv(CovB1A)*CovB2B1A';
    GABVec(ID)=sum(log(eig(Sigma1)))-sum(log(eig(Sigma2)));
end
TABVec=real(TABVec);
GABVec=real(GABVec);
TAB=mean(TABVec);
GAB=mean(GABVec);

