



% fs - sampling frequency, Rsub - number of trials preserved, winT - epoch
% duration as number of samples, Xclean2 - preprocessed EEG epochs, 
% AmpsMclean - MEP amplitudes (see help lines in the function)
[y, xtildef]=...
    labelDataMEPs2(Xclean2, AmpsMclean, winT, Rsub,fs);

% hyperparameters to be cross-validated for testing the final accuracy 
nDynVec=[2 4 8 20]';
regulCSPVec=[1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2]';

testAccFinal=[];
WcollSpatAll=[];
%
n=5; % 5 folds in cross validation (outer loop)
for i1=1:n

    %choose test trials by their indices
    iTest=isortMEP([i1:n:25*1, (length(isortMEP)-(25*1-i1)):n:length(isortMEP)]);
    % training trial indices
    iTr0=false(1,numTrials);
    % choose the training trials
    iTr0(isortMEP([1:150, (length(isortMEP)-149):length(isortMEP)]))=true;
    %exclude test trials
    iTr0(iTest)=false;

    CVlist=setdiff(1:n,i1); %use the rest folds for inner loop of nested CV


    for iReg=1:length(regulCSPVec)

        kCV=1; %indices run from 1 to 4 in the inner loop
        for i1CV=CVlist

            %choose CV indices
            iCV=isortMEP([i1CV:n:25*1, (length(isortMEP)-(25*1-i1CV)):n:length(isortMEP)]);
            % inner loop training set incides
            iTr=iTr0;
            % exclude CV indices from the training set
            iTr(iCV)=false;

            [WcollSpat]=CSPforSpatialFilter(xCorr(:,:,iTr), y(iTr), ...
                regulCSPVec(iReg), xCorr(:,:,iTr));
            WcollSpatAll(:,:,iReg, kCV)=WcollSpat;

            [score, testAcc]=getTestScoreVersion1_4(xCorr, iTr, y, iCV, ...
                WcollSpat, 3, nDynVec(3), 1, true,[], true,[]);
            testAccAll(iReg, kCV)=testAcc;
            kCV=kCV+1;
        end
    end
    %choose maximum accuracy -giving regul. parameter
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
            [score, testAcc]=getTestScoreVersion1_4(xCorr, iTr, y, iCV, ...
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

    [WcollSpat]=CSPforSpatialFilterTEP(xCorr, y(iTr), regulCSPVec(iRegMax), xCorr);
    % st scoring
    [score, testAcc]=getTestScoreVersion1_4(xCorr, iTr, y, iTest, WcollSpat, ...
        3, nDynVec(iDynMax), 1, true,[],true,[]);

    testAccFinal(i1)=testAcc;

end
%final result
Accuracy= mean(testAccFinal)
