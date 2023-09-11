clc;clear;close all;
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
addpath(genpath([rootpath,'steinmetz-et-al-2019-master\']))
addpath(genpath([rootpath,'npy-matlab-master\']))

%% ------------------------------------------------------------------------
subdir='total_clusterArea_paper';
rootpath='..\dataset\Mice\result\';
datapath=[rootpath,subdir];
outpath=[datapath,'_results_kernel_0\'];
clusterRegPath=[rootpath,'clusterArea_paper'];
if(~exist(outpath,'dir'))
    mkdir(outpath);
end

load(clusterRegPath)
regCluster=0;% use clusters of regions
filename='BasicInfo';
dir_info=dir(datapath);
winSize=0.001;
dt=0.001;
config=[];
config.fixLength=1;
config.stimTimelock=0;
config.stimBoundary=[1,.5];%is used when isTimelock=1
config.wheelTimelock=1;
config.wheelBoundary=[.5,0.3];%is used when isTimelock=1
config.stimEpoch=0;%is used when isTimelock=0
config.baseline=0;
config.baseBoundary=[1,0];

spikeArea={};
file_postfix=['_trlFixLen_',num2str(config.fixLength),'_dt_1'];

sess=1;

unitCount=[];
for i=1 : 41
    if(dir_info(i).isdir==0 || strcmp(dir_info(i).name,'.') || strcmp(dir_info(i).name,'..'))
        continue;
    end
    
    filepath=[datapath,'\',dir_info(i).name];
    dataPath2=[filepath,'\',filename];
    load(dataPath2);
    outpath2=[outpath,'\',dir_info(i).name];
    if(~exist(outpath2,'dir'))
        mkdir(outpath2);
    end

    trlNo=size(trialInfo,1);
    
    spikeTimesStimLock={};
    stimBounday={};
    for j=1 : trlNo
        trl=j;
        for k=1 : length(clusterSpikeTimes)
            % ------------------- fix length trials -----------------------
            if(config.fixLength)
                if(config.stimTimelock)
                    stimBounday=[-config.stimBoundary(1),config.stimBoundary(2)];
                    boundary=[trialInfo.stimOn(trl)-config.stimBoundary(1),trialInfo.stimOn(trl)+config.stimBoundary(2)];
                    idx=find(clusterSpikeTimes{k}>=boundary(1) & clusterSpikeTimes{k}<=boundary(2));
                    if(~isempty(idx))
                        spikeTimesStimLock{j,k}=clusterSpikeTimes{k}(idx)-trialInfo.stimOn(trl);
                    else
                        spikeTimesStimLock{j,k}=[];
                    end
                end

                if(config.wheelTimelock)
                    wheelBounday=[-config.wheelBoundary(1),config.wheelBoundary(2)];
                    boundary=[trialInfo.wheelOn(trl)-config.wheelBoundary(1),trialInfo.wheelOn(trl)+config.wheelBoundary(2)];
                    idx=find(clusterSpikeTimes{k}>=boundary(1) & clusterSpikeTimes{k}<=boundary(2));
                    if(~isempty(idx) && ~isnan(trialInfo.wheelOn(trl)))
                        spikeTimesWheelLock{j,k}=clusterSpikeTimes{k}(idx)-trialInfo.wheelOn(trl);
                    else
                        spikeTimesWheelLock{j,k}=[];
                    end
                end

            else
                % ------------------ free length trials ------------------
                if(config.stimEpoch) % boundary: [stim,response]
                    wh=trialInfo.stimOn(trl)+.3;
                    if(~isnan(trialInfo.wheelOn(trl)))
                        wh=trialInfo.wheelOn(trl);
                    end
                    stimBounday{j}=[trialInfo.stimOn(trl)-config.stimBoundary(1),wh+config.wheelBoundary(2)]-trialInfo.stimOn(trl);
                    boundary=[trialInfo.stimOn(trl)-config.stimBoundary(1),wh+config.wheelBoundary(2)];
                    idx=find(clusterSpikeTimes{k}>=boundary(1) & clusterSpikeTimes{k}<=boundary(2));
                    if(~isempty(idx))
                        spikeTimesStimLock{j,k}=clusterSpikeTimes{k}(idx)-trialInfo.stimOn(trl);
                    else
                        spikeTimesStimLock{j,k}=[];
                    end
                end
            end
        end
    end

    %% spikeTime
    processedArea=unique(unitInfo.area);
    if(config.stimTimelock || config.stimEpoch)
        stimSpikeRate=SpikeRate(spikeTimesStimLock,unitInfo.area,stimBounday,dt);
        save([outpath2,'\stimSpikeRates',file_postfix],'config','dt','stimBounday','winSize','stimSpikeRate','processedArea','-v7.3');
    end
    
    if(config.wheelTimelock)
        wheelSpikeRate=SpikeRate(spikeTimesWheelLock,unitInfo.area,wheelBounday,dt);
        save([outpath2,'\wheelSpikeRates',file_postfix],'config','dt','wheelBounday','winSize','wheelSpikeRate','processedArea','-v7.3');
    end

    invalid_idx=strcmp(processedArea,'invalid');
    processedArea(invalid_idx)=[];
    for c=1 : length(processedArea)
        len=length(spikeArea);
        idx=find(strcmp(spikeArea,processedArea{c}));
        if(isempty(idx))
            area_idx=len+1;
        else
            area_idx=idx(1);
        end
        spikeArea{area_idx}=processedArea{c};
    end
 
    disp(num2str(sess))
    sess=sess+1;
end

save([outpath,'areaInfo'],'spikeArea','unitCount')
