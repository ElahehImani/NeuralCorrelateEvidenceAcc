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

alpha=0.05;
type='contra';%cp,dp,contra
oneSide=0;
all_units=1;

switch(type)
    case 'evidence'
        oneSide=0;
    case 'contra'
        oneSide=1;
    case 'dp'
        oneSide=1;
    case 'choice'
        oneSide=0;
end

stimAligned=1;
if(stimAligned)
    filepostfix='_stim';
    selectivityBoundary=[0,0.3];
else
    filepostfix='_wheel';
    selectivityBoundary=[-0.3,0];
end

rootpath=['..\dataset\Mice\spike\result\',subdir];
outpath=[rootpath,'_results_kernel_0\'];

respath=[outpath,'\_auroc\'];


if(all_units)
    filepostfix=[filepostfix,'_allUnits'];
end


roc=load([respath,'temporal_',type,filepostfix]);
roc_chance=load([respath,'temporal_',type,filepostfix,'_chance']);
load([outpath,'areaInfo'])
decodingRes_h=cell(length(spikeArea),39);
validIndex=cell(length(spikeArea),39);
idx1=find(roc.times>=selectivityBoundary(1));
idx2=find(roc.times>=selectivityBoundary(2));
boundary=idx1(1) : idx2(1);

%% processing
for i=1 : length(spikeArea)
    for j=1 : 39
        if(isempty(roc.decodingRes{i,j}))
            continue;
        end
        
        [unitCount,smpl]=size(roc.decodingRes{i,j});
        decodingRes_h{i,j}=zeros(unitCount,smpl);
        if(~isfield(roc,'validIndex'))
            validIndex{i,j}=1:unitCount;
        end
        
        for u=1 : unitCount
            [~,I]=max(abs(roc.decodingRes{i,j}(u,boundary)-0.5));
            I=I+boundary(1)-1;
            for t=1 : smpl
                roc_val=squeeze(roc_chance.decodingRes{i,j}(:,u,t));
                meanRoc=nanmean(roc_val);
                CI=computeCI(roc_val,'percentile',alpha);
                if(~oneSide)
                    if(~isnan(roc.decodingRes{i,j}(u,t)) && (roc.decodingRes{i,j}(u,t)<CI(1) || roc.decodingRes{i,j}(u,t)>CI(2)))
                        decodingRes_h{i,j}(u,t)=1;
                    end
                else
                    if(~isnan(roc.decodingRes{i,j}(u,t)) && roc.decodingRes{i,j}(u,t)>CI(2))
                        decodingRes_h{i,j}(u,t)=1;
                    end
                end

            end
        end
    end
    disp(['area ',num2str(i)])
end

decodingRes=roc.decodingRes;
times=roc.times;
if(isfield(roc,'validIndex'))
    validIndex=roc.validIndex;
end

save([respath,'temporal_',type,filepostfix],'times','decodingRes','decodingRes_h','validIndex')

