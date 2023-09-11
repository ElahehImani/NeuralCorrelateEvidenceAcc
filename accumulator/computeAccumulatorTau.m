clc;clear;close all
%% ------------------------------------------------------------------------
rootpath='..\dataset\Accumulator\spikes_subsample_postWheel\';
method={'single_collaps','race_collaps','compete_collaps'};
regions={'frontal','striatum','visual','midbrain','MOpSSp','thalamus','midbrain_striatum'};
maxLag=0.3;
lag=0.025;

for r=3 : length(regions)
    for m=1 : length(method)
        filepath=[rootpath,regions{r},'\_results_',method{m},'\_timescaleData\'];
        respath=[rootpath,regions{r},'\_results_',method{m},'\_tau\'];
        figPath=[respath,'_images\'];
        files=dir(fullfile(filepath,'*.mat'));
        
        if(length(files)==0)
            continue;
        end
        
        if(~exist(respath,'dir'))
            mkdir(respath);
        end
        if(~exist(figPath,'dir'))
            mkdir(figPath);
        end

        for f=1 : length(files)
            nameParts=split(files(f).name,'_');
            load([filepath,files(f).name]);
            [fitR,autocorrData]=getAccTau(result,lag,maxLag);
            tau=fitR.tau;
            autocorr=nanmean(autocorrData,1);

            h=figure;hold on;
            lag_num=round(maxLag/(lag))-1;
            lags=(1:lag_num)*lag*1000;
            plot(lags,autocorr,'bo','markerFaceColor','b');
            plot(lags,fitR.modelFunc(fitR.expFitParams,lags*1e-3), '-','linewidth',1,'Color','k');
            text(lags(end-7),.3,['\tau = ',num2str(1000*fitR.tau,'%.2f'),' ms'])

            save([respath,nameParts{1},'_tau.mat'],'tau','autocorr');
            saveas(h,[figPath,nameParts{1},'_tau.pdf'])
            close(h);
        end
    end
end



