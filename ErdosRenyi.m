function G=ErdosRenyi(N,P)

%%
% With these two parameters, we can instantiate the graph. The 
% variable |G| is the adjacency matrix for the graph. However,
% the first step doesn't treat edges symmetrically. The last
% two operations fix this and yield a symmetric adjacency matrix.

rand('seed',100); % reseed so you get a similar picture
G = rand(N,N) < P;
G = triu(G,1);
G = G + G'; 