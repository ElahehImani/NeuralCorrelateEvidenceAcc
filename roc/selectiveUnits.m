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

rootpath=['..\dataset\Mice\result\',subdir];
outpath=[rootpath,'_results_kernel_0\'];
respath=[outpath,'\_auroc\'];

clusterThresh=3;
type='cp';%cp,dp,contra

all_units=1;

stimAligned=0;
if(stimAligned)
    filepostfix='_stim';
else
    filepostfix='_wheel';
end


switch(type)
    case {'choice','evidence'}
        if(stimAligned)
            selectivityBoundary=[0,0.3];
        else
            selectivityBoundary=[-0.3,0];
        end
        oneSided=0;
        
    case 'contra'
        if(stimAligned)
            selectivityBoundary=[0,0.2];
        else
            selectivityBoundary=[-0.25,-0.05];
        end
        oneSided=1;
        
    case 'dp'
        if(stimAligned)
            selectivityBoundary=[0,0.3];
        else
            selectivityBoundary=[-0.3,0];
        end
        oneSided=1;
end

samplingRate=100;

if(all_units)
    filepostfix=[filepostfix,'_allUnits'];
end


ROC=load([respath,'temporal_',type,filepostfix]);
load([outpath,'areaInfo'])

respath=[respath,'selectiveUnits\'];
if(~exist(respath,'dir'))
    mkdir(respath);
end

validIndex=cell(length(spikeArea),39);
idx1=find(ROC.times>=selectivityBoundary(1));
idx2=find(ROC.times>=selectivityBoundary(2));
bound=idx1(1) : idx2(1);

selective=cell(length(spikeArea),39);
selectiveAmp=cell(length(spikeArea),39);
selectiveTimes=cell(length(spikeArea),39);
%% processing
for i=1 : length(spikeArea)
    for j=1 : 39
        if(isempty(ROC.decodingRes{i,j}))
            continue;
        end
        
        [unitCount,smpl]=size(ROC.decodingRes{i,j});
        sig=sigClusterAnalysis(ROC.decodingRes_h{i,j}(:,bound),clusterThresh);
        selective{i,j}=zeros(unitCount,1);
        selectiveAmp{i,j}=NaN(unitCount,1);
        selectiveTimes{i,j}=NaN(unitCount,1);
        rocTimes=ROC.times(bound);
        for u=1 : unitCount
            realRoc=ROC.decodingRes{i,j}(u,bound);
            
            roc=(realRoc-0.5);
            roc(sig(u,:)==0)=NaN;
%             roc=sig(u,:).*(realRoc-0.5);
            if(oneSided)
                [v,I]=nanmax(roc);
                if(isnan(v))
                    continue;
                end
                if(realRoc(I)>0.5)
                    selective{i,j}(u)=1;
                    selectiveAmp{i,j}(u)=realRoc(I);
                    selectiveTimes{i,j}(u)=rocTimes(I);
                end
            else
                [v,I]=nanmax(abs(roc));
                if(isnan(v))
                    continue;
                end
                if(realRoc(I)>0.5)
                    selective{i,j}(u)=1;
                    selectiveAmp{i,j}(u)=realRoc(I);
                    selectiveTimes{i,j}(u)=rocTimes(I);
                elseif(realRoc(I)<0.5)
                    selective{i,j}(u)=-1;
                    selectiveAmp{i,j}(u)=realRoc(I);
                    selectiveTimes{i,j}(u)=rocTimes(I);
                end
            end
        end
    end
    
    disp(['area ',num2str(i)])
end

save([respath,'selectiveUnits_',type,filepostfix],'selective','selectiveAmp','selectiveTimes','clusterThresh','selectivityBoundary')


