function Xfilt=filter3DTempSpat(X,wtemp,wspat, standardize)
% X: data epochs (channels x times x trials
% wtemp: temporal filter
% wspat: spatial filter
% standardize: just put false

[C, T, R]=size(X);
if ~isempty(wspat)
    NCompS=size(wspat,2);
XfiltS=reshape(wspat'*reshape(X, C,R*T),NCompS, T, R);
else
    XfiltS=X;
    NCompS=C;
end

if ~isempty(wtemp)
    NCompT=size(wtemp,2);
    if ~standardize
        Xfilt=permute(reshape(wtemp'*reshape(permute(XfiltS,[2 1 3]),[T,NCompS*R]),NCompT,NCompS,R),[2 1 3]);
    else
        for it=1:NCompT
            wtempIt=wtemp(:,it);
            wtempIL=sum(wtempIt==0);
            
            XfiltSI=XfiltS./repmat(sqrt(mean(mean(X(:,wtempIL+1:end,:).^2,1),2)),[size(XfiltS,1) size(XfiltS,2) 1]);
            Xfilt(:,it,:)=permute(reshape(wtempIt'*reshape(permute(XfiltSI,[2 1 3]),[T,NCompS*R]),1,NCompS,R),[2 1 3]);
            
        end
    end
else
    Xfilt=XfiltS;
    
end