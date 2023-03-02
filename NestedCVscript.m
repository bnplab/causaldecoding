



% fs - sampling frequency, Rsub - number of trials preserved, winT - epoch
% duration as number of samples, Xclean2 - preprocessed EEG epochs, 
% AmpsMclean - MEP amplitudes (see help lines in the function)
[y, xtildef]=...
    labelDataMEPs2(Xclean2, AmpsMclean, winT, Rsub,fs);
numTrials=length(y);
isortMEP=1:numTrials;

% hyperparameters to be cross-validated for testing the final accuracy 
nDynVec=[2 4 8 20]'; % number of features for ML training
regulCSPVec=[1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2]'; %regularization coefficient for CSP

testAccFinal=[];
WcollSpatAll=[];
%
n=5; % 5 folds in cross validation (outer loop for final accuracy testing)
for i1=1:n

    %choose test trials by their indices, here 25 largest and smallest
    iTest=isortMEP([i1:n:25*1, (length(isortMEP)-(25*1-i1)):n:length(isortMEP)]);
    % training trial indices
    iTr0=false(1,numTrials);
    % choose the training trials, here amont the 150 largest and smallest
    iTr0(isortMEP([1:150, (length(isortMEP)-149):length(isortMEP)]))=true;
    %exclude test trials
    iTr0(iTest)=false;

    CVlist=setdiff(1:n,i1); %use the remaining folds for inner loop of nested CV

    % computing CSP filters for the inner loop
    for iReg=1:length(regulCSPVec)

        kCV=1; %indices run from 1 to 4 in the inner loop
        for i1CV=CVlist

            %choose CV indices, again among the 25 largest and smallest MEPs
            iCV=isortMEP([i1CV:n:25*1, (length(isortMEP)-(25*1-i1CV)):n:length(isortMEP)]);
            % inner loop training set incides are the same as in the outer loop, ...
            iTr=iTr0;
            % but excluding CV indices
            iTr(iCV)=false;

            [WcollSpat]=CSPforSpatialFilter(xtildef(:,:,iTr), y(iTr), ...
                regulCSPVec(iReg), xtildef(:,:,iTr), [], false, 3, 0);
            WcollSpatAll(:,:,iReg, kCV)=WcollSpat; %save the spatial filters to save time
            %The accuracies are computed for 8 features with various regularization coeffs
            [score, testAcc]=getTestScoreVersion1_4(xtildef, iTr, y, iCV, ...
                WcollSpat, 3, nDynVec(3), 1, true,[], true,[]);
            testAccAll(iReg, kCV)=testAcc;
            kCV=kCV+1;
        end
    end
    %choose maximum accuracy-giving regularization parameter, and use this in below to determin the optimal number of features (in nDynVec)
    [~, iRegMax]=max(mean(testAccAll,2));

    for iDyn=1:length(nDynVec)
        kCV=1; %indices run from 1 to 4 in the inner loop
        for i1CV=CVlist

            %choose CV indices
            iCV=isortMEP([i1CV:n:25*1, (length(isortMEP)-(25*1-i1CV)):n:length(isortMEP)]);
            % inner loop training set incides
            iTr=iTr0;
            % exclude CV indices from the training set
            iTr(iCV)=false;
            [score, testAcc]=getTestScoreVersion1_4(xtildef, iTr, y, iCV, ...
                squeeze(WcollSpatAll(:,:,iRegMax, kCV)), 3, nDynVec(iDyn), 1, true,[], true,[]);
            testAccAll(iDyn, kCV)=testAcc;
            kCV=kCV+1;
        end
    end
    [~, iDynMax]=max(mean(testAccAll,2));
    close all
    pause(.1)

    %outer loop testing
    iTr=iTr0;

    [WcollSpat]=CSPforSpatialFilter(xtildef(:,:,iTr), y(iTr), regulCSPVec(iRegMax), xtildef(:,:,iTr),...
        , [], false, 3, 0));
    % getting test accuracy for outer loop round i1
    [score, testAcc]=getTestScoreVersion1_4(xtildef, iTr, y, iTest, WcollSpat, ...
        3, nDynVec(iDynMax), 1, true,[],true,[]);

    testAccFinal(i1)=testAcc;

end
%final result
Accuracy= mean(testAccFinal)
