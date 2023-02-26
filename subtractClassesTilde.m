function Cov=subtractClassesTilde(var_tilde, N, y, K)



[C,R]=size(var_tilde);
Cov=zeros(C,C);
Xdiff=zeros(C, floor(sum(y)/N),floor(sum(~y)/N));
for k=1:K
    
XmeansTrue=randomMeansTilde(var_tilde(:,y), N);
XmeansFalse=randomMeansTilde(var_tilde(:,~y), N);

for i=1:size(XmeansTrue,3)
    for j=1:size(XmeansFalse,3)
        Xdiff(:,i,j)=XmeansTrue(:,i)-XmeansFalse(:,j)*1;
    end
end
Cov=Cov+reshape(Xdiff,C,[])*reshape(Xdiff,C,[])'/size(Xdiff,2)/K;

end