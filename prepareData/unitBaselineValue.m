clc;clear;close all
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%%
baselineName='baselineData_kernel_0';
rootpath='..\dataset\Mice\spike\result\total_results_kernel_0\';

filename='stimSpikeRates_trlFixLen_1';
load([rootpath,'areaInfo']);
outpath=[rootpath,'_baselineData\'];
if(~exist(outpath,'dir'))
    mkdir(outpath);
end

samplingRate=100;
outfileName=baselineName;

baseline_boundary=[-0.9,-0.1];
baselineData=cell(length(spikeArea),39);

sess=1;
for i=1 : 41
    if(strcmp(dirInfo(i).name,'.') || strcmp(dirInfo(i).name,'..'))
        continue;
    end
    
    load([rootpath,dirInfo(i).name,'\',filename]);
    if(isempty(stimSpikeRate))
        sess=sess+1;
        continue;
    end
    times=stimBounday(1) : dt : stimBounday(2);
    if(length(times)>size(stimSpikeRate{1,1},2))
        times=times(1)+dt/2 : dt : times(end);
    end
    dt=times(2)-times(1);
    sr=1/dt;
    if(binned)
        step=round(sr/samplingRate);
        sample_idx=1:step:length(times);
    else
        sample_idx=1:length(times);
    end
    
    aidx=0;
    for k=1 : length(processedArea)
        if(~isempty(find(strcmp(processedArea{k},'invalid'))))
            continue;
        end
        aidx=aidx+1;
        
        area=find(strcmp(processedArea{k},spikeArea));
        
        trlNo=size(stimSpikeRate,1);
        unitNo=size(stimSpikeRate{1,aidx},1);
        
        avgR=NaN(trlNo,unitNo);
        spikeC=NaN(trlNo,unitNo);
        for u=1 : unitNo
            for t=1 : trlNo
                spikes=stimSpikeRate{t,aidx}(u,:);
                
                if(binned)
                    spikes = hist(times(find(spikes)), times(sample_idx));
                end
                
                fr=spikes; 
                idx1=find(times(sample_idx)>=baseline_boundary(1));
                idx2=find(times(sample_idx)>=baseline_boundary(2));
                boundary=idx1(1):idx2(1);
                
                avgR(t,u)=nanmean(fr(boundary));
                spikeC(t,u)=sum(spikes(boundary));
            end
        end
        
        data=[];
        data.avgRate=avgR;
        data.spikeCount=spikeC;
        data.trialNo=trlNo;
        data.unitNo=unitNo;
        baselineData{area,sess}=data;
    end
    
    disp(sess)
    sess=sess+1;
    save([outpath,outfileName],'baselineData','baseline_boundary');
end

