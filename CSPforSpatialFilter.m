function [WcollSpat, chKeep]=CSPforSpatialFilter(xtildefTrain, yTrain, ...
    gamma, Xclean2F, penaltyC, useLasso, nS, nCkeep)
%
% This function computes spatial filters using the training data and the trial labels 
%
% input:
%
% xtildefTrain - train data set (channels x time points x trials)
% yTrain - labels of the epochs
% gamma - regularization constant, e.g., [1e-4 ,1e3] 
% Xclean2F - insert here the same as xtildefTrain. Train data set (separately because could include data from all 
% recording not just the epochs prior to TMS)
% penaltyC - covariance matrix to be used in regularization (if
% noise covariance is knows). Use [], if prior penalty is used.
% useLasso - true if you wish to completely remove channels from spatial
% filter estimation
% nCkeep - number of channels to keep
%
% output:
%
% WcollSpat: spatial filters as columns (eigen values from CSP), (channels x channels)


[C, Nsub, ~]=size(xtildefTrain);
chKeep=1:C;
CovAllSpat=zeros(C, C, Nsub);
for i=1:Nsub
    
    CovTemp=subtractClassesTilde(reshape(xtildefTrain(:,i,:),C,[]), 25, yTrain,400);%400
    CovAllSpat(:,:,i)=CovTemp;%./trace(CovTemp);
end

%CovNAllSpat=reshape(xtildefTrain(:,:,:),C,[]);

CovNAllSpat=reshape(Xclean2F(:,:,:),C,[]);
CovNAllSpat=CovNAllSpat*CovNAllSpat'/size(CovNAllSpat,2);
if isempty (penaltyC)
    [W,D]=eig(sum(CovAllSpat(:,:,:),3),sum(CovNAllSpat,3)*1+...
    eye(C)*trace(sum(CovNAllSpat(:,:,:),3))*gamma/C);

if useLasso
    chKeepB=true(C,1);
    for nCkeepI=(C-1):-1:nCkeep
       
    [~ , isort]=sort(real(diag(D)),'descend');
    Aspat=sum(CovAllSpat(chKeepB,chKeepB,:),3)*W(:,isort(1:nS));
    varCh=min(abs(Aspat.*W(:,isort(1:nS))),[],2);
    [~ , isort]=sort(varCh,'descend');
    indsRemain=find(chKeepB);
    indsRemove=indsRemain(isort(end));
    chKeepB(indsRemove)=false;
    
    [W,D]=eig(sum(CovAllSpat(chKeepB,chKeepB,:),3),sum(CovNAllSpat(chKeepB,chKeepB,:),3)*1+...
    eye(sum(chKeepB))*trace(sum(CovNAllSpat(chKeepB,chKeepB,:),3))*gamma/sum(chKeepB));
    end
    chKeep=(chKeepB);
end


else
[W,D]=eig(sum(CovAllSpat(:,:,:),3),sum(CovNAllSpat,3)*1+...
    eye(C)*trace(sum(CovNAllSpat(:,:,:),3))*gamma*0/C+...
    penaltyC*trace(sum(CovNAllSpat(:,:,:),3))*gamma/trace(penaltyC));
end
[~ , isort]=sort(real(diag(D)),'descend');
Wcoll=W(:,isort);
WcollSpat=Wcoll./repmat(sqrt(sum(Wcoll.^2,1)),[size(Wcoll,1),1]);