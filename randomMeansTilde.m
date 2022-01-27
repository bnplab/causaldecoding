function Xmeans=randomMeansTilde(X, N)
[C, R]=size(X);
irand=randperm(R);
Xmeans=zeros(C,floor(R/N));
for i=1:floor(R/N)
    
    Xmeans(:,i)=mean(X(:,irand((i-1)*N+1:N*i)),2);
end