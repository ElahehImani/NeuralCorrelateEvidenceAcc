clc;clear;close all;
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%% settings --------------------------------------------------------------
subdir='total';
regCluster=1;
clusterName='clusterArea_paper';
if(regCluster)
    subdir=[subdir,'_',clusterName];
end

rootpath='..\dataset\Mice\result\';
ratePath=[rootpath,subdir,'_results_kernel_20_half_gaussian\'];
load([rootpath,'total_clusterArea_paper_results_kernel_0\_FiringRate\spikeCount_go.mat'])
behavPath=[rootpath,subdir];
outpath=[ratePath,'_clustering\'];
baselineData=[ratePath,'_baselineData\'];
all_units=1;
postfix='_allUnits';

if(~exist(outpath,'dir'))
    mkdir(outpath);
end

fixLength=1;
file_postfix=['_trlFixLen_',num2str(fixLength)];


samplingRate=200;

baselineCorrection=1;

baselineData=[baselineData,'baselineData_kernel_0.mat'];
load(baselineData);

baseinfoFilename='BasicInfo';
areaInfo='areaInfo';
load([ratePath,areaInfo]);

featureName={'L0-R0','L0-R0-go','L0-R0-nogo','L0-R0-right','L0-R0-left',...
                    'L0-R25','L0-R25-go','L0-R25-nogo','L0-R25-right','L0-R25-left',...
                    'L0-R50','L0-R50-go','L0-R50-nogo','L0-R50-right','L0-R50-left',...
                    'L0-R100','L0-R100-go','L0-R100-nogo','L0-R100-right','L0-R100-left',...
                    'L25-R0','L25-R0-go','L25-R0-nogo','L25-R0-right','L25-R0-left',...
                    'L25-R25','L25-R25-go','L25-R25-nogo','L25-R25-right','L25-R25-left',...
                    'L25-R50','L25-R50-go','L25-R50-nogo','L25-R50-right','L25-R50-left',...
                    'L25-R100','L25-R100-go','L25-R100-nogo','L25-R100-right','L25-R100-left',...
                    'L50-R0','L50-R0-go','L50-R0-nogo','L50-R0-right','L50-R0-left',...
                    'L50-R25','L50-R25-go','L50-R25-nogo','L50-R25-right','L50-R25-left',...
                    'L50-R50','L50-R50-go','L50-R50-nogo','L50-R50-right','L50-R50-left',...
                    'L50-R100','L50-R100-go','L50-R100-nogo','L50-R100-right','L50-R100-left'...
                    ,'L100-R0','L100-R0-go','L100-R0-nogo','L100-R0-right','L100-R0-left',...
                    'L100-R25','L100-R25-go','L100-R25-nogo','L100-R25-right','L100-R25-left',...
                    'L100-R50','L100-R50-go','L100-R50-nogo','L100-R50-right','L100-R50-left',...
                    'L100-R100','L100-R100-go','L100-R100-nogo','L100-R100-right','L100-R100-left',...
                    'go','nogo','right','left','failed','missed',...
                    'R=0','R=0-go','R=0-nogo','R=0-right','R=0-left',...
                    'R=25','R=25-go','R=25-nogo','R=25-right','R=25-left',...
                    'R=50','R=50-go','R=50-nogo','R=50-right','R=50-left',...
                    'R=100','R=100-go','R=100-nogo','R=100-right','R=100-left'};
                
dir_info=dir(behavPath);
contrastLevels=[1,.75,.5,.25,0,-.25,-.5,-.75,-1];
cont=[0,.25,.5,1];
count=1;

trialNum=cell(length(spikeArea),length(featureName));
sessIdx=cell(1,length(spikeArea));
unitIdx=cell(1,length(spikeArea));
SubArea=cell(1,length(spikeArea));
features=cell(length(spikeArea),length(featureName));
TimeBoundary_stim=[-0.1,0.3];
TimeBoundary_wheel=[-0.3,0.1];
sess=1;

featureName=featureName';

for i=1 : 41
    if(dir_info(i).isdir==0 || strcmp(dir_info(i).name,'.') || strcmp(dir_info(i).name,'..'))
        continue;
    end

    load([behavPath,'\',dir_info(i).name,'\',baseinfoFilename]);
    load([ratePath,'\',dir_info(i).name,'\stimSpikeRates',file_postfix]);
    load([ratePath,'\',dir_info(i).name,'\wheelSpikeRates',file_postfix]);
    subArea=unitInfo.subArea;
    Area=unitInfo.area;
    times_stim=stimBounday(1) : dt : stimBounday(2);
    if(size(stimSpikeRate{1,1},2)<length(times_stim))
        times_stim=times_stim(1)+dt/2 : dt : times_stim(end);
    end
    
    times_wheel=wheelBounday(1) : dt : wheelBounday(2);
    if(size(wheelSpikeRate{1,1},2)<length(times_wheel))
        times_wheel=times_wheel(1)+dt/2 : dt : times_wheel(end);
    end
    
    sr=1/dt;

     sample_idx_stim=1:length(times_stim);
     sample_idx_wheel=1:length(times_wheel);
    
    idx1=find(times_stim(sample_idx_stim)>=TimeBoundary_stim(1));
    idx2=find(times_stim(sample_idx_stim)>=TimeBoundary_stim(2));
    boundary_stim=idx1(1) : idx2(1);
    
    idx1=find(times_wheel(sample_idx_wheel)>=TimeBoundary_wheel(1));
    idx2=find(times_wheel(sample_idx_wheel)>=TimeBoundary_wheel(2));
    boundary_wheel=idx1(1) : idx2(1);
    
    reg=0;
    for k=1 : length(processedArea)
        if(~isempty(find(strcmp(processedArea{k},'invalid'))))
            continue;
        end
        
        ridx=find(strcmp(Area,processedArea{k}));
        reg=reg+1;
        regIdx=find(strcmp(spikeArea,processedArea{k}));

        totalUnitCount=size(stimSpikeRate{1,reg},1);
        validIdx=find(avgFiringRate{regIdx,sess}>=1)';
        
        sessIdx{regIdx}=[sessIdx{regIdx};sess*ones(length(validIdx),1)];
        unitIdx{regIdx}=[unitIdx{regIdx};validIdx];
        SubArea{regIdx}=[SubArea{regIdx};subArea(ridx(validIdx))];
        
        if(isempty(validIdx))
            continue;
        end
        
        t=1;
        spikeActivity=[];
        cond=[];
                
        choice=trialInfo.chosenResponse;
        RT=trialInfo.wheelOn-trialInfo.stimOn;
        right=trialInfo.rightContrastLevel;
        left=trialInfo.leftContrastLevel;
        evidence=right-left;
        correct=trialInfo.correct;
        activity=[];
        activity_wheel=[];
        for j=1 : length(trialInfo.correct)
            fr_stim=stimSpikeRate{j,reg}(validIdx,boundary_stim);
            fr_wheel=wheelSpikeRate{j,reg}(validIdx,boundary_wheel);
            
            if(baselineCorrection)
                for u=1 : length(validIdx)
                    if(nanmean(baselineData{regIdx,sess}.spikeCount(:,validIdx(u)))>0)
                        meanB=nanmean(baselineData{regIdx,sess}.avgRate(:,validIdx(u)));
                        stdB=nanstd(baselineData{regIdx,sess}.avgRate(:,validIdx(u)));
                        fr_stim(u,:)=(fr_stim(u,:)-meanB)./stdB;
                        fr_wheel(u,:)=(fr_wheel(u,:)-meanB)./stdB;
                    else
                        fr_stim(u,:)=NaN;
                        fr_wheel(u,:)=NaN;
                    end
                end
            end
            
            activity(j,:,:)=fr_stim;
            activity_wheel(j,:,:)=fr_wheel;
        end
        
        validTrl=(choice==0) | (choice~=0 & ~isnan(RT));
        c=1;
        for l=1 : 4
            for r=1 : 4
                tidx=validTrl & cont(l)==left & cont(r)==right;
                f=NaN(length(validIdx),length(boundary_stim));
                if(sum(tidx)==0)
                    features{regIdx,c}=[features{regIdx,c};f];
                    trialNum{regIdx,c}=[trialNum{regIdx,c};zeros(length(validIdx),1)];
                else
                    if(size(activity,2)>1)
                        features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(tidx,:,:),1))];
                    else
                        features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(tidx,:,:),1))'];
                    end
                    trialNum{regIdx,c}=[trialNum{regIdx,c};sum(tidx)*ones(length(validIdx),1)];
                end
                c=c+1;
                
                Go=tidx & choice~=0;
                if(size(activity,2)>1)
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Go,:,:),1))];
                else
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Go,:,:),1))'];
                end
                trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Go)*ones(length(validIdx),1)];
                c=c+1;

                NoGo=tidx & choice==0;
                if(size(activity,2)>1)
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(NoGo,:,:),1))];
                else
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(NoGo,:,:),1))'];
                end
                trialNum{regIdx,c}=[trialNum{regIdx,c};sum(NoGo)*ones(length(validIdx),1)];
                c=c+1;

                Right=tidx & choice==-1;
                if(size(activity,2)>1)
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Right,:,:),1))];
                else
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Right,:,:),1))'];
                end
                trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Right)*ones(length(validIdx),1)];
                c=c+1;

                Left=tidx & choice==1;
                if(size(activity,2)>1)
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Left,:,:),1))];
                else
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Left,:,:),1))'];
                end
                trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Left)*ones(length(validIdx),1)];
                c=c+1;
            end
        end
        
        
        Go=validTrl & choice~=0;
        if(size(activity,2)>1)
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Go,:,:),1))];
        else
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Go,:,:),1))'];
        end
        trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Go)*ones(length(validIdx),1)];
        c=c+1;
        
        NoGo=validTrl & choice==0;
        if(size(activity,2)>1)
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(NoGo,:,:),1))];
        else
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(NoGo,:,:),1))'];
        end
        trialNum{regIdx,c}=[trialNum{regIdx,c};sum(NoGo)*ones(length(validIdx),1)];
        c=c+1;
        
        Right=validTrl & choice==-1;
        if(size(activity,2)>1)
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Right,:,:),1))];
        else
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Right,:,:),1))'];
        end
        trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Right)*ones(length(validIdx),1)];
        c=c+1;
        
        Left=validTrl & choice==1;
        if(size(activity,2)>1)
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Left,:,:),1))];
        else
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Left,:,:),1))'];
        end
        trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Left)*ones(length(validIdx),1)];
        c=c+1;

        failed=validTrl & choice~=0 & correct==0;
        if(size(activity,2)>1)
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(failed,:,:),1))];
        else
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(failed,:,:),1))'];
        end
        trialNum{regIdx,c}=[trialNum{regIdx,c};sum(failed)*ones(length(validIdx),1)];
        c=c+1;
        
        missed=validTrl & choice==0 & correct==0;
        if(size(activity,2)>1)
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(missed,:,:),1))];
        else
            features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(missed,:,:),1))'];
        end
        trialNum{regIdx,c}=[trialNum{regIdx,c};sum(missed)*ones(length(validIdx),1)];
        c=c+1;
        
        
        for r=1 : 4
            tidx=validTrl & cont(r)==right;
            f=NaN(length(validIdx),length(boundary_stim));
            if(sum(tidx)==0)
                features{regIdx,c}=[features{regIdx,c};f];
                trialNum{regIdx,c}=[trialNum{regIdx,c};zeros(length(validIdx),1)];
            else
                if(size(activity,2)>1)
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(tidx,:,:),1))];
                else
                    features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(tidx,:,:),1))'];
                end
                trialNum{regIdx,c}=[trialNum{regIdx,c};sum(tidx)*ones(length(validIdx),1)];
            end
            c=c+1;
            
            Go=tidx & choice~=0;
            if(size(activity,2)>1)
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Go,:,:),1))];
            else
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Go,:,:),1))'];
            end
            trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Go)*ones(length(validIdx),1)];
            c=c+1;
            
            NoGo=tidx & choice==0;
            if(size(activity,2)>1)
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(NoGo,:,:),1))];
            else
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(NoGo,:,:),1))'];
            end
            trialNum{regIdx,c}=[trialNum{regIdx,c};sum(NoGo)*ones(length(validIdx),1)];
            c=c+1;
            
            Right=tidx & choice==-1;
            if(size(activity,2)>1)
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Right,:,:),1))];
            else
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Right,:,:),1))'];
            end
            trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Right)*ones(length(validIdx),1)];
            c=c+1;
            
            Left=tidx & choice==1;
            if(size(activity,2)>1)
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Left,:,:),1))];
            else
                features{regIdx,c}=[features{regIdx,c};squeeze(nanmean(activity(Left,:,:),1))'];
            end
            trialNum{regIdx,c}=[trialNum{regIdx,c};sum(Left)*ones(length(validIdx),1)];
            c=c+1;
        end
        
    end
    
    times_stim=times_stim(sample_idx_stim);
    times=times_stim(boundary_stim);
    
    disp(['sess', num2str(sess)])
    sess=sess+1;
end

save([outpath,'clusteringFeature',postfix],'features','spikeArea','featureName','trialNum','times','sessIdx','unitIdx','SubArea','-v7.3')

