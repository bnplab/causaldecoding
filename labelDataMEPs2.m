function [y, xtildef, mepdelta, iR, Xclean2F, mepdeltaAll, b]=...
    labelDataMEPs2(Xclean2, AmpsMclean, winT, Rsub,fs)

% This function labels the EEG epochs and chooses the most reliably
% labelled trials
%
% input:
% Xlean2: EEG data (Channels x times x trials)
% AmpsMclean: Mep amplitudes (Trials x EMG channels)
% winT: how many  samples to be taken into the epoch (counting from the
% right end to the left)
% Rsub: how many epochs to include in output
% fs: sampling frequency for filtering. 0 if no filtering
%
% output:
% y: (Rsub x 1) class labels for each trial (true, false: high mep, low mep)
% xtildef: (Channels x winT x Rsub) data epochs as requested by inputs
% mepdelta: (Rsub x 1) combined mep amplitude (geometric mean) difference to trendline
% iR: indices of mepdelta from the original AmpsMclean
% Xclean2F: filtered data (not needed)
% mepdeltaAll: all mep differences (not needed)
% b: trendlines

%log10 scaling
mepsize=log10(AmpsMclean);

for iC=1:size(mepsize,2)
    %mirror-extension of MEPs at the beginning and end. Median filter to detect
    %the trend
    btemp=max(medfilt1([flipud(mepsize(:,iC)); mepsize(:,iC); ...
        flipud(mepsize(:,iC))],20, [],1), log10(thAmp));
    b(:,iC)=btemp((size(mepsize,1)+1):(end-size(mepsize,1)));

end

y=mepsize>(b); %larger or smaller that trendline
mepdelta=mepsize-b*1; %detrended mep amplitudes

% filter if desired with an FIR filter
if fs~=0

    highF=45;
    bf = fir1(50, highF./(fs*.5));
    Xclean2F=permute(filtfilt(bf, 1,permute(Xclean2,[2 1 3])),[2 1 3]);
else
    Xclean2F=Xclean2;
end

iR=1:size(mepdelta,1); %trial indices
if size(mepdelta,2)>1
    % joint amplitude of MEPs
    mepdelta=sum(abs(mepdelta.^2),2).*sign(mean(mepdelta,2));
    iEq=~(all(y,2) | all(~y,2));% indices where detrended amplitudes have opposite signs
    %remove them from trial indices
    iR(iEq)=[];
end
%sort detrended amplitudes
[~,isort]=sort(mepdelta(iR),'descend');
isort=iR(isort); %indices from original data

n=round(Rsub/2); %number of low-excitability/high-excitability trials
iR=[isort(1:n) isort(end-n+1:end)];
mepdeltaAll=mepdelta;

mepdelta=mepdelta(iR); %sort and extract Rsub detrended trials 
xtildef=double(Xclean2F(:,end-winT:end,iR)); %extract EEG trials
y=y(iR,1); %extract labels for the trials

[C, Nsub, R]=size(xtildef);
xtildef=xtildef-repmat(mean(xtildef,2),[1 Nsub 1]); %center the trials