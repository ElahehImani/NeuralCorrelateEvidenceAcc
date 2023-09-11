clc;clear;%close all;
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
addpath(genpath([root,'steinmetz-et-al-2019-master']))
%% settings --------------------------------------------------------------
subdir='total';
clusterName='clusterArea_paper';
regCluster=1;
if(regCluster)
    subdir=[subdir,'_',clusterName];
end
rootpath=['..\dataset\Mice\spike\result\',subdir];
% outpath=[rootpath,'_results_kernel_20_half_gaussian\'];
outpath=[rootpath,'_results_kernel_0\'];
respath=[outpath,'\_firingRate\'];

bin_width=0.05;
RT_thresh=[0.15,1];
clusterThresh=5;
fixedLen=1;
trialType='total';%right,left,total
out_postfix='';
if(fixedLen)
    out_postfix='_fixedLen';
end

stimAligned=1;
if(stimAligned)
    out_postfix=[out_postfix,'_stim_go'];
else
    out_postfix=[out_postfix,'_wheel_go'];
end

if(~isempty(RT_thresh))
    out_postfix=[out_postfix,'RT_',num2str(1000*RT_thresh(1)),'_',num2str(1000*RT_thresh(2))];
end

% out_postfix=[out_postfix,'_kernel_20_half_gaussian'];
out_postfix=[out_postfix,'_bin_',num2str(1000*bin_width)];
filename=['latencyAuroc',out_postfix];

out_postfix=[out_postfix,'_cluster',num2str(clusterThresh)];

switch(trialType)
    case {'right','left'}
        out_postfix=[out_postfix,'_',trialType];
end

outfilename=['latency',out_postfix];
load([outpath,'areaInfo']);
load([respath,filename]);
latency=cell(length(spikeArea),39);
dt=times(2)-times(1);
for i=1 : length(spikeArea)
    for j=1 : 39
        if(isempty(rateAuroc{i,j}))
            continue;
        end
        
        ucount=size(rateAuroc{i,j},1);
        latency{i,j}=NaN(ucount,1);
        for u=1 : ucount
            sig_roc=rateAuroc_h{i,j}(u,:);
            auroc=rateAuroc{i,j}(u,:);
            sig_roc=sig_roc & auroc>0.5;
            sig_roc=sigClusterAnalysis(sig_roc,clusterThresh);
            lat=find(sig_roc);
            
%             lat = find(filter(ones(1,clusterThresh),1,sig_roc) == clusterThresh & auroc>0.5,1);
            if(~isempty(lat))
                lat=lat(1);
                latency{i,j}(u)=dt*lat;
            end
        end
    end
end

save([respath,outfilename],'latency')

