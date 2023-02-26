function [score, testAcc, FitInfo]=getTestScoreVersion1_4(xtildef, trainIdx, ...
    y, testIdx, WcollSpat, nCompS, nDyn, version, normalize, Af, doCV, gammaLasso)
%
% This function computes accuracies of ST decoding for test trials after
% classifier with training trials. The
% spatial filters need to be precomputed (using training trials).
% input: 
%
% xtildef - 3D EEG data matrix (channels x time points x trials)
% trainIdx - boolean valued indices for trainin trials (1 x trials), true =
% training trial, false = not used in training
% y - labels (0 or false = low excitability or class A, and true or 1 = high
% excitability  or class B)
% testIdx - test trial indices as integers (1 x test trials)
% WcollSpat - From CSP the spatial filters as columns
% nCompS - number of used spatial filters, i.e., WcollSpat(:, 1:nCompS) 
% nDyn - number of features used in classification
% version - use 1
% normalize - use 1 if using prestimulus EEG (and if power is not to be used as a classification criterion) and 0 if TEPs
% Af - addition classification features, use []
% doCV - true = use crossValidation to determine the regularization parameter by
% CV. Otherwise, compute accuracies with several regularization parameters
% gammaLasso - vector of regularization parameters for lassoglm function.
% If [], use the default set: logspace(-3,-1,15);
%
% output:
% score - logistic regression scores to be used in classification (for test
% trials)
% testAcc - accuracies for test trials. As long a vector as gammaLasso if doCV = false 
% FitInfo - FittingInfo struct from lassoglm function
%
%

[~, Nsub, ~]=size(xtildef);
Xdec=filter3DTempSpat(xtildef(:,:,trainIdx),[],WcollSpat(:,1:nCompS), false);


if isempty(gammaLasso)
    gammaLasso=logspace(-2,-0,15)*1e-1;
end

switch version
    case 1
        CovST=subtractClassesTilde(reshape(Xdec,[nCompS*Nsub,sum(trainIdx)]), 25, y(trainIdx),2000);
    case 4
        CovST=reshape(Xdec,[nCompS*Nsub,sum(trainIdx)]);
        CovST=CovST*CovST'/size(CovST,2);
end


[WST, ~ ,~]=svds(CovST, nDyn);%,


dynComp=WST(:,1:nDyn)'*reshape(filter3DTempSpat(xtildef,[],WcollSpat(:,1:nCompS), false),nCompS*Nsub,[]);%

if normalize
    
    dynComp=dynComp./sqrt(sum(dynComp.^2));
   
end
if ~isempty(Af)
    dynComp=[dynComp; Af];
end

if doCV
rng('default')
[B,FitInfo] = lassoglm(dynComp(:,trainIdx)',y(trainIdx),'binomial', 'alpha', ...
    1e-5, 'Options',statset('UseParallel',true),'lambda', ...%alpha .1 alpha 1
    gammaLasso,'standardize',1, 'cv', 5);%1e0 1e-1 (coeff)
figure
lassoPlot(B,FitInfo,'PlotType','CV');
idx=FitInfo.IndexMinDeviance;
Bmin=B(:,idx);
B0 = FitInfo.Intercept(idx);
score=Bmin'*dynComp(:,testIdx)+B0;

CM=[sum(score>0 & y(testIdx)') sum(score>0 & ~y(testIdx)');...
                        sum(score<=0 & y(testIdx)') sum(score<=0 & ~y(testIdx)')];
                    

testAcc=sum(diag(CM))/sum(CM(:));
else
    rng('default')
    
[B,FitInfo] = lassoglm(dynComp(:,trainIdx)',y(trainIdx),'binomial', 'alpha', ...
    1e-5, 'Options',statset('UseParallel',true),'lambda', ...%alpha .1
    gammaLasso,'standardize',1);%1e0 1e-1 (coeff)


scoreAll=B'*dynComp(:,:)+FitInfo.Intercept';%(FitInfo.Index1SE);

score=scoreAll(:,testIdx);%B'*Sd(1:end,iTest)+FitInfo.Intercept(FitInfo.IndexMinDeviance);
CM=[permute(sum(score>0 & y(testIdx)',2),[2 3 1]) permute(sum(score>0 & ~y(testIdx)',2),[2 3 1]);...
    permute(sum(score<=0 & y(testIdx)',2),[2 3 1]) permute(sum(score<=0 & ~y(testIdx)',2),[2 3 1])];

testAcc=squeeze((CM(1,1,:)+CM(2,2,:))./sum(sum(CM,1),2))';%trace(CM)/sum(CM(:))

end


