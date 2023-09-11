clc;clear;close all
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%%
subdir='total';
clusterName='clusterArea_paper';
regCluster=1;
if(regCluster)
    subdir=[subdir,'_',clusterName];
end

all_units=1;
postfix='';
if(all_units)
    postfix='_allUnits';
end

rootpath='..\dataset\Mice\spike\result\';
ratepath=[rootpath,subdir,'_results_kernel_0\'];
behavPath=[rootpath,subdir,'\'];
choicePath=[rootpath,subdir,'_results_kernel_20_half_gaussian\_auroc\selectiveUnits\finalSelective_clustering_dp_visualGrading',postfix];

includePostWheel=1;
postWheelLen=0.05;
outpath=[rootpath,'Accumulator\spikes_totalUnit_postWheel\'];

if(~exist(outpath,'dir'))
    mkdir(outpath);
end

load([ratepath,'areaInfo']);
load([tauPath,tauFile]);
load([tauPath,gradingFile]);
load(choicePath);

dirInfo=dir(behavPath);

RT_boundary=[.15,0.5];
unitThresh=5;%min number of units per session
latency_boundary=[-0.5,0.5];
lat_bin_width=0.1;
clusterThresh=2;
bootsrap_count=zeros(1,8);
sess=1;
for i=1 : length(dirInfo)
    if(strcmp(dirInfo(i).name,'.') || strcmp(dirInfo(i).name,'..'))
        continue;
    end
    
    load([behavPath,dirInfo(i).name,'\BasicInfo']);
    if(includePostWheel)
        load([ratepath,dirInfo(i).name,'\stimSpikeRates_trlFixLen_0_dt_1_postWheel']);
    else
        load([ratepath,dirInfo(i).name,'\stimSpikeRates_trlFixLen_0_dt_1']);
    end
    
    fixedLenData=load([ratepath,dirInfo(i).name,'\stimSpikeRates_trlFixLen_1_dt_1']);
    
    times_1=fixedLenData.stimBounday(1) : fixedLenData.dt : fixedLenData.stimBounday(2);
    
    idx1=find(times_1>=latency_boundary(1));
    idx2=find(times_1<=latency_boundary(2));
    bnd=idx1(1) : idx2(end);
    
    RT_total=trialInfo.wheelOn-trialInfo.stimOn;
    right_total=trialInfo.rightContrastLevel;
    left_total=trialInfo.leftContrastLevel;
    choice_total=trialInfo.chosenResponse;
    
    validTrl=(~isnan(RT_total) & choice_total~=0);
    if(~isempty(RT_boundary))
        validTrl=validTrl & RT_total>=RT_boundary(1) & RT_total<=RT_boundary(2);
    end
    
    c=0;
    for j=1 : length(processedArea)
        if(strcmp(processedArea{j},'invalid'))
            continue;
        end
        
        c=c+1;
        if(strcmp(processedArea{j},'other'))
            continue;
        end
        
        aidx=find(strcmp(spikeArea,processedArea{j}));
        
        validUnits=find(finalSelective{aidx,sess});
        if(length(validUnits)<unitThresh)
            continue;
        end
        
        % --------------------------- spec area ---------------------------
        outpath2=[outpath,spikeArea{aidx},'\'];
        if(~exist(outpath2,'dir'))
            mkdir(outpath2);
        end
        
        unitIdx=validUnits;

        %% compute latency
        trls=find(validTrl);
        spikes=[];
        lat_times=[];
        for t=1 : length(trls)
            spikes{t}=fixedLenData.stimSpikeRate{trls(t),c}(unitIdx,bnd);
            lat_times{t}=times_1(bnd);
        end
        
        bin_width=round(lat_bin_width/fixedLenData.dt);
        [auroc,sig_roc] = latency_roc_popLevel(spikes,lat_times,bin_width,lat_times{1}(end));
        
        sig_roc=sig_roc & auroc>0.5;
        sig_roc=sigClusterAnalysis(sig_roc,clusterThresh);
        sig_roc=find(sig_roc);
        lat=NaN;
        if(~isempty(sig_roc))
            lat=sig_roc(1);
        end
        
        if(isnan(lat))
            continue;
        end
        
        %% select trials with RT higher than latency
        validTrl2=validTrl & RT_total>=(lat*fixedLenData.dt+0.05) & right_total~=left_total;
        trls=find(validTrl2);
        spikes=[];
        times=[];
        for t=1 : length(trls)
            spikes{t}=stimSpikeRate{trls(t),c}(unitIdx,:);
            times{t}=stimBounday{trls(t)}(1) : dt : stimBounday{trls(t)}(end);
        end
        
        unitTau=fitData{aidx,sess}.tau(unitIdx);
        popTau=fitR.tau;
        latency_ms=lat*dt;
        latency_smpl=lat;
        bootsrap_count(aidx)=bootsrap_count(aidx)+1;
        RT=RT_total(trls);
        right=right_total(trls);
        left=left_total(trls);
        choice=choice_total(trls);
        
        save([outpath2,'pop',num2str(bootsrap_count(aidx))],'sess','unitIdx','spikes','times',...
            'latency_ms','latency_smpl','RT','right','left','choice','postWheelLen')
    end
    sess=sess+1;
end

