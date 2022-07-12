%% Define networks
% Define network A: a Watts-Strogatz network
WA= WattsStrogatz(300,15,0.50);
WA = full(adjacency(WA,'weighted'));
WeightBaseA=(1+rand(size(WA,1),size(WA,2))*9);
WA=WA.*(triu(WeightBaseA,1)+triu(WeightBaseA,1)');

% Define network B: a Erdos-Renyi network
WB= ErdosRenyi(500,0.2);
WeightBaseB=(1+rand(size(WB,1),size(WB,2))*9);
WB=WB.*(triu(WeightBaseB,1)+triu(WeightBaseB,1)');

%% Represent nerworks as Gaussian Markov random fields
Type=1; % Define the covariance matrix as L+1/n J
[LA,PinvLA,SigmaA]=GMRandomField(WA,Type);
[LB,PinvLB,SigmaB]=GMRandomField(WB,Type);

%% Calculate information diverigence between nerworks
% Network approximation if two networks have different sizes
[NLA,NLB,NPinvLA,NPinvLB,NSigmaA,NSigmaB,SortLEbasedC,Index,LE,NLE,Gamma]=NetworkApproximation(WA1,WB,LA,LB,PinvLA,PinvLB,SigmaA,SigmaB);
% Calculate diverigence
[DAB,DBA]=InfoDivergence(NSigmaA,NSigmaB);

%% Calculate mutual information between nerworks
SampleNum=5000;
K=2;
[HA,HB,HAB,IAB]=MutualInfo(SigmaA,SigmaB,SampleNum,K);

%% Calculate Fisher information between nerworks
% Here we define a case of decoding, where there exists a parameter vector
% controlled by network B. The actual state of this parameter vector can
% affect network A. We calcualte Fisher information matrix as a reflection
% of the information of network B contained in network A. Please see our
% paper for more detailed explanations.

% Step 1: Generate a matrix of the observation of parameter vector, where
% each row is a unique observation. Here the parameter vector is defined
% following the approach introduced in Sec. 6 in our paper
DegNewNetwork=sum(WB);
Numberoftheta=10;
NumberofO=100;
ThetaMatrix=zeros(NumberofO,Numberoftheta);
for ID=1:NumberofO
    RandomIDB=randperm(size(WB,1));
    ThetaMatrix(ID,:)=DegNewNetwork(RandomIDB(1:Numberoftheta));
end
ThetaMatrix=unique(ThetaMatrix,'rows');

% Step 2: Assume that the effect of parameter vector on network A is to
% add a noise on the Gaussian Markov random field of network A. Here the
% definition of the noise is introduced in Sec. 6 in our paper
SigmaEnsemble=zeros(NumberofO,size(SigmaA,1),size(SigmaA,2));
for ID=1:NumberofO
    WeightBase=abs(normrnd(0,mean(ThetaMatrix(ID,:)),size(SigmaA,1),size(SigmaA,2)));
    [~,~,NoiseSigmaA1]=GMRandomField(WA1.*(triu(WeightBase,1)+triu(WeightBase,1)'),Type);
    SigmaEnsemble(ID,:,:)=NoiseSigmaA1;
end

% Step 3: Calculate the Fisher information matrix
[FIMatrix]=FisherInfo(SigmaEnsemble,ThetaMatrix);

%% Calculate Granger causality and transfer entropy between two networks
RandomPNum=20;
[TABVec,TAB,GABVec,GAB,SizeABVec]=GrangerCandTransferE(SigmaA,SigmaB,SampleNum,RandomPNum,K);
[TBAVec,TBA,GBAVec,GBA,SizeBAVec]=GrangerCandTransferE(SigmaB,SigmaA,SampleNum,RandomPNum,K);

