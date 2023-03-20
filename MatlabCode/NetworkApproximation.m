function [NLA,NLB,NPinvLA,NPinvLB,NSigmaA,NSigmaB,SortLEbasedC,Index,LE,NLE,Gamma]=NetworkApproximation(WA,WB,LA,LB,PinvLA,PinvLB,SigmaA,SigmaB)
% This function implements network approximation if the two networks have
% different sizes (contain different numbers of nodes). The approximation
% is defined based on Laplacian energy. 

% Input: 
% WA: the weighted adjacent matrix of network A
% WB: the weighted adjacent matrix of network B
% LA: the Laplacian of network A
% LB: the Laplacian of network B
% PinvLA: the Moore每Penrose pseudoinverse of LA
% PinvLB: the Moore每Penrose pseudoinverse of LB
% SigmaA: the covariance matrix Sigma of network A
% SigmaB: the covariance matrix Sigma of network B

% Output:
% NLA: new Laplacian matrix of network A. If network A has a larger size,
% then NLA \neq LA because it will be modified by approximation. Otherwise,
% NLA=LA.
% NLB: new Laplacian matrix of network B. If network B has a larger size,
% then NLB \neq LB because it will be modified by approximation. Otherwise,
% NLB=LB.
% NPinvLA: the Moore每Penrose pseudoinverse of NLA
% NPinvLB: the Moore每Penrose pseudoinverse of NLB
% NSigmaA: the new covariance matrix of Gaussian Markov random field of
% network A
% NSigmaB: the new covariance matrix of Gaussian Markov random field of
% network B
% Gamma: the rationality of approximation

if size(WA,1)==size(WB,1) % These two networks have the same size and approximation is unnecessary
   NLA=LA;
   NLB=LB;
   NPinvLA=PinvLA;
   NPinvLB=PinvLB; 
   NSigmaA=SigmaA;
   NSigmaB=SigmaB;
   SortLEbasedC=zeros(size(WA,1),1);
   Index=zeros(size(WA,1),1);
   LE=0;
   NLE=0;
   Gamma=1;
elseif size(WA,1)>size(WB,1) % We need to do approximation for network A
   % Calculate Laplacian energy 
   DegA=sum(WA).^2;
   WAwithZeroDiag=WA-diag(diag(WA));
   LE=sum(DegA)+sum(sum(WAwithZeroDiag.^2));
   % Calculate Delta Laplacian energy 
   DeltaLE=zeros(size(WA,1),1);
   WA2=WA*WA;
   WA2withZeroDiag=WA2-diag(diag(WA2));
   for ID=1:size(WA,1)
       % The adjacent matrix without this node
       MA=WA;
       MA(ID,:)=[];
       MA(:,ID)=[];
       MA2=MA*MA;
       MA2withZeroDiag=MA2-diag(diag(MA2));
       % Measure Delta Laplacian energy of this node8
       DeltaLE(ID)=4*WA2(ID,ID)+2*(sum(sum(WA2withZeroDiag))-sum(sum(MA2withZeroDiag)));
   end
   LEbasedC=DeltaLE./LE;
   [SortLEbasedC,Index] = sort(LEbasedC); 
   NeededNodes=Index(end-size(WB,1)+1:end); % Find k=size(WB,1) nodes with relatively large Laplacian centrality 
   WA=WA(NeededNodes,NeededNodes);
   DegA = sum(WA);
   NLA=diag(DegA)-WA;
   NLB=LB;
   NPinvLA=PinvFunction(NLA);
   NPinvLB=PinvLB;
   NSigmaA=NLA+ones(size(NPinvLA,1),size(NPinvLA,1)).*1/size(NPinvLA,1);
   NSigmaB=SigmaB;
   % Calculate the rationality of approximation
   DegAS=sum(WA).^2;
   WAwithZeroDiag=WA-diag(diag(WA));
   NLE=sum(DegAS)+sum(sum(WAwithZeroDiag.^2));
   Gamma=NLE/LE;
elseif size(WA,1)<size(WB,1) % We need to do approximation for network B
   % Calculate Laplacian energy 
   DegB=sum(WB).^2;
   WBwithZeroDiag=WB-diag(diag(WB));
   LE=sum(DegB)+sum(sum(WBwithZeroDiag.^2));
   % CBlculBte DeltB LBplBciBn energy 
   DeltaLE=zeros(size(WB,1),1);
   WB2=WB*WB;
   WB2withZeroDiag=WB2-diag(diag(WB2));
   for ID=1:size(WB,1)
       % The adjacent matrix without this node
       MB=WB;
       MB(ID,:)=[];
       MB(:,ID)=[];
       MB2=MB*MB;
       MB2withZeroDiag=MB2-diag(diag(MB2));
       % Measure Delta Laplacian energy of this node
       DeltaLE(ID)=4*WB2(ID,ID)+2*(sum(sum(WB2withZeroDiag))-sum(sum(MB2withZeroDiag)));
   end 
   LEbasedC=DeltaLE./LE; 
   [SortLEbasedC,Index] = sort(LEbasedC);
   NeededNodes=Index(end-size(WA,1)+1:end); % Find k=size(WA,1) nodes with relatively large Laplacian centrality 
   WB=WB(NeededNodes,NeededNodes);
   DegB = sum(WB);
   NLB=diag(DegB)-WB;
   NLA=LA;
   NPinvLB=PinvFunction(NLB);
   NPinvLA=PinvLA;
   NSigmaB=NLB+ones(size(NPinvLB,1),size(NPinvLB,1)).*1/size(NPinvLB,1);
   NSigmaA=SigmaA;
   % Calculate the rationality of approximation
   DegBS=sum(WB).^2;
   WBwithZeroDiag=WB-diag(diag(WB));
   NLE=sum(DegBS)+sum(sum(WBwithZeroDiag.^2));
   Gamma=NLE/LE;
end