clc;clear;close all;
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%% settings --------------------------------------------------------------
subdir='total';
clusterName='clusterArea_paper';
regCluster=1;
if(regCluster)
    subdir=[subdir,'_',clusterName];
end

rootpath=['..\dataset\Mice\result\',subdir];
ratepath=[rootpath,'_results_kernel_0\'];
outpath=[rootpath,'_results_kernel_0\'];
respath=[outpath,'\_firingRate\'];
out_postfix='_go';
fixLength=0;
file_postfix=['_trlFixLen_',num2str(fixLength),'_dt_1'];
% file_postfix=['_trlFixLen_',num2str(fixLength)];
activityFilename=['stimSpikeRates',file_postfix];

if(~exist(respath,'dir'))
    mkdir(respath);
end

baseinfoFilename='BasicInfo';
areaInfo='areaInfo';
load([outpath,areaInfo]);

dir_info=dir(rootpath);

avgFiringRate=cell(length(spikeArea),39);

%% conditions
sess=1;
for i=1 : 41
    if(dir_info(i).isdir==0 || strcmp(dir_info(i).name,'.') || strcmp(dir_info(i).name,'..'))
        continue;
    end
    
    load([rootpath,'\',dir_info(i).name,'\',baseinfoFilename]);
    load([ratepath,'\',dir_info(i).name,'\',activityFilename]);
    if(isempty(stimSpikeRate))
        sess=sess+1;
        continue;
    end
   
    reg=0;
    for k=1 : length(processedArea)
        if(~isempty(find(strcmp(processedArea{k},'invalid'))))
            continue;
        end
        reg=reg+1;
        regIdx=find(strcmp(spikeArea,processedArea{k}));
        for t=1 : size(stimSpikeRate,1)
            totalUnitCount=size(stimSpikeRate{t,reg},1);
            if(totalUnitCount~=0)
                break;
            end
        end
        
        
        rate=NaN(1,totalUnitCount);
        choice=trialInfo.chosenResponse;
        RT=trialInfo.wheelOn-trialInfo.stimOn;
        right=trialInfo.rightContrastLevel;
        left=trialInfo.leftContrastLevel;
        validRT=ones(size(RT));
        trlIdx=find(choice~=0 & ~isnan(RT) & right~=left & validRT);
        
        for u=1 : totalUnitCount
            spikeCount=0;
            for t=1 : length(trlIdx)
                if(isempty(stimSpikeRate{trlIdx(t),reg}))
                    continue;
                end
                spikeCount=spikeCount+sum(stimSpikeRate{trlIdx(t),reg}(u,:));
            end
            rate(u)=spikeCount/length(trlIdx);
        end
        
        avgFiringRate{regIdx,sess}=rate;
    end
    
    save([respath,'spikeCount',out_postfix],'avgFiringRate')
    disp(['sess', num2str(sess)])
    sess=sess+1;
end

