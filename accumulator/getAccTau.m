function [fitR,totalAutocorr]=getAccTau(result,lag,maxLag)
shuffel=length(result);
totalAutocorr=[];
for s=1 : shuffel
    Y=result{s}.Y;
    times=0.001:0.001:maxLag;
    bins=0:lag:maxLag;
    Lags=bins(2:end-1);
    [trlNo,smpl,unitNum]=size(Y);
    trialBins=[];
    for u=1 : unitNum
        for t=1 : trlNo
            trialBins(u,t,:)=get_data_bin(squeeze(Y(t,:,u)),bins,times);
        end
    end

    maxlag_idx = round(maxLag/lag);
    for u=1 : unitNum
        corr_data=cell(1,maxlag_idx-1);
        count=zeros(1,maxlag_idx-1);
        acorr=zeros(unitNum,length(corr_data));
        for i=1 : length(bins)-1
            for j=i+1 : length(bins)-1
                dBin = j - i;
                if (var(trialBins(u,:,i)) == 0) || (var(trialBins(u,:,j)) == 0)
                    binCorr = 0;
                else
                    binCorr = corr(trialBins(u,:,i)',trialBins(u,:,j)');
                end

                corr_data{dBin}=[corr_data{dBin},binCorr];
                count(dBin) = count(dBin) + 1;
            end
        end
        
        acorr=[];
        for cc=1 : length(corr_data)
            acorr(cc) = nanmean(corr_data{cc});
        end

        totalAutocorr=[totalAutocorr;acorr];
    end
end

y=totalAutocorr;
x=repmat(Lags,size(totalAutocorr,1),1);
fitR=fit_tau(y(:),x(:),maxLag,lag);
end

%%
function [bin_data]=get_data_bin(x,bin,times)
x=abs(x);
if(size(x,1)>1)
    [max_x,I]=max(max(x,[],2),[],1);
    x=x(I(1),:);
end

bin_data=[];
for i=1 : length(bin)-1
    start=find(times>=bin(i));
    start=start(1);
    last=find(times>=bin(i+1));
    last=last(1);

    bin_data(i)=nanmean(x(start:last));
end
end