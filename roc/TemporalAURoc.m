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

type='evidence';% cp,dp,evidence
all_units=1;

stimAligned=1;
if(stimAligned)
    filepostfix='_stim';
    activityFilename='stimSpikeRates';
    activityVarName='stimSpikeRate';
    boundaryVarName='stimBounday';
    TimeBoundary=[-0.1,0.3];
else
    filepostfix='_wheel';
    activityFilename='wheelSpikeRates';
    activityVarName='wheelSpikeRate';
    boundaryVarName='wheelBounday';
    TimeBoundary=[-0.3,0.1];
end

rootpath=['..\dataset\Mice\spike\result\',subdir];
outpath=[rootpath,'_results_kernel_0\'];

baselineData=[outpath,'_baselineData\'];
respath=[outpath,'\_auroc\'];

fixLength=1;
file_postfix=['_trlFixLen_',num2str(fixLength)];

if(all_units)
    filepostfix=[filepostfix,'_allUnits'];
end

baselineCorrection=1;
if(baselineCorrection)
    baselineData=[baselineData,'baselineData_kernel_0'];
    load(baselineData);
end

if(~exist(respath,'dir'))
    mkdir(respath);
end

baseinfoFilename='BasicInfo';
areaInfo='areaInfo';
load([outpath,areaInfo]);

dir_info=dir(rootpath);
cutoff=5;
evidenceLevels=[1,.75,.5,.25,0,-.25,-.5,-.75,-1];
contrastLevels=[0,.25,.5,1];
cond=[];
count=1;
for i=1 : length(contrastLevels)
    for j=1 : length(contrastLevels)
        ContrastCond(count,:)=[contrastLevels(i),contrastLevels(j)];
        count=count+1;
    end
end
choiceAlt=[-1,0,1];

condNo=size(ContrastCond,1);
decodingRes=cell(length(spikeArea),39);
validIndex=cell(length(spikeArea),39);

%% conditions
sess=1;
for i=1 : 41
    if(dir_info(i).isdir==0 || strcmp(dir_info(i).name,'.') || strcmp(dir_info(i).name,'..'))
        continue;
    end

    load([rootpath,'\',dir_info(i).name,'\',baseinfoFilename]);
    a=load([outpath,'\',dir_info(i).name,'\',activityFilename,file_postfix]);
    boundary=a.(boundaryVarName);
    spikeRate=a.(activityVarName);
    dt=a.dt;
    processedArea=a.processedArea;
    if(isempty(spikeRate))
        sess=sess+1;
        continue;
    end
    times=boundary(1) : dt : boundary(2);
    if(size(spikeRate{1,1},2)<length(times))
        times=times(1)+dt/2 : dt : times(end);
    end

    sample_idx=1:length(times);
    idx1=find(times(sample_idx)>=TimeBoundary(1));
    idx2=find(times(sample_idx)>=TimeBoundary(2));
    boundary2=idx1(1) : idx2(1);

    reg=0;
    cond=zeros(length(trialInfo.correct),1);
    for k=1 : length(processedArea)
        reg=reg+1;
        regIdx=find(strcmp(spikeArea,processedArea{k}));
        totalUnitCount=size(spikeRate{1,reg},1);
        validIdx=1:totalUnitCount;
        spikeActivity=[];
        choice=trialInfo.chosenResponse;
        correctness=trialInfo.correct;
        RT=trialInfo.wheelOn-trialInfo.stimOn;
        right=trialInfo.rightContrastLevel;
        left=trialInfo.leftContrastLevel;
        right_evidence=right-left;

        for j=1 : length(trialInfo.correct)
            spikeRates=spikeRate{j,reg}(validIdx,:);
            fr=spikeRates(:,boundary2);
            if(baselineCorrection)
                for u=1 : length(validIdx)
                    if(nanmean(baselineData{regIdx,sess}.spikeCount(:,validIdx(u)))>0)
                        meanB=nanmean(baselineData{regIdx,sess}.avgRate(:,validIdx(u)));
                        stdB=nanstd(baselineData{regIdx,sess}.avgRate(:,validIdx(u)));
                        fr(u,:)=(fr(u,:)-meanB)./stdB;
                    else
                        fr(u,:)=NaN;
                    end
                end
            end

            spikeActivity(j,1:length(validIdx),:)=fr;

            cond(j)=find(ContrastCond(:,1)==trialInfo.leftContrastLevel(j) & ContrastCond(:,2)==trialInfo.rightContrastLevel(j));
        end

        %% decoding
        [~,unitNo,smpl]=size(spikeActivity);
        cp=nan(unitNo,length(boundary2));
        decodingRes{regIdx,sess}=NaN(totalUnitCount,length(boundary2));
        validTrl=choice==0 | (choice~=0 & ~isnan(RT));
        trlIndex=cell(1,16);
        trlIndex_null=cell(1,16);

        switch(type)
            case 'cp'
                validTrl=validTrl & right~=left & choice~=0;
                for c=1 : 16
                    trlIndex{c}=find(validTrl & correctness==1 & cond==c);
                    trlIndex_null{c}=find(validTrl & correctness==0 & cond==c);
                end

            case 'dp'
                validTrl=validTrl & right~=left;
                for c=1 : 16
                    trlIndex{c}=find(validTrl & choice~=0 & correctness==1 & cond==c);
                    trlIndex_null{c}=find(validTrl & choice==0 & cond==c);
                end
            
            case 'choice'
                validTrl=validTrl & choice~=0;
                for c=1 : 16
                    trlIndex{c}=find(validTrl & choice==-1 & cond==c);
                    trlIndex_null{c}=find(validTrl & choice==1 & cond==c);
                end

            case 'evidence'
                validTrl=validTrl & correctness==1;
                evidenceLevels2=evidenceLevels;
                evidenceLevels2(evidenceLevels2==0)=[];
                for c=1 : length(evidenceLevels2)-1
                    trlIndex{c}=find(validTrl & right_evidence==evidenceLevels2(c));
                    for c2=c+1 : length(evidenceLevels2)
                        trlIndex_null{c}=[trlIndex_null{c};find(validTrl & right_evidence==evidenceLevels2(c2))];
                    end
                end
        end

        parfor j=1 : unitNo
            for s=1 : smpl
                nTotal = 0;
                dTotal = 0;
                for c=1 : condNo
                    nA=length(trlIndex{c});
                    nB=length(trlIndex_null{c});
                    if(nA+nB<cutoff || nA==0 || nB==0)
                        continue;
                    else
                        spike=squeeze(spikeActivity(trlIndex{c},j,s));
                        spike_null=squeeze(spikeActivity(trlIndex_null{c},j,s));
                        if(sum(isnan(spike))>0 || sum(isnan(spike_null))>0)
                            continue;
                        end
                        nTotal = nTotal + (nA*nB)*rocM(spike_null,spike);
                        dTotal = dTotal+nA*nB;
                    end
                end

                if(dTotal>0)
                    cp(j,s) = nTotal./dTotal;
                end
            end
        end

        decodingRes{regIdx,sess}(validIdx,:)=cp;
        validIndex{regIdx,sess}=validIdx;
    end

    times=times(sample_idx);
    times=times(boundary2);
    save([respath,'temporal_',type,filepostfix],'times','decodingRes','validIndex','-v7.3')

    disp(['sess', num2str(sess)])
    sess=sess+1;
end

