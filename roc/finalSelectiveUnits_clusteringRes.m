clc;clear;close all
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%%
rootpath='..\dataset\Mice\result\';
rootpath2=[rootpath,'total_clusterArea_paper_results_kernel_0\'];
selectivePath=[rootpath2,'_auroc\selectiveUnits\'];
dpcaPath=[rootpath,'total_clusterArea_paper_results_kernel_0\_clustering\_dpca\stim_dp'];
all_units=1;
file_postfix='';
if(all_units)
    file_postfix=[file_postfix,'_allUnits'];
end

dpcaPath=[dpcaPath,file_postfix,'\clustersInfo'];
dpca=load(dpcaPath);

firingRatePath=[rootpath,'total_clusterArea_paper_results_kernel_0\_FiringRate\'];

evidence=load([selectivePath,'selectiveUnits_evidence_stim',file_postfix]);
choice=load([selectivePath,'selectiveUnits_choice_wheel',file_postfix]);
dp=load([selectivePath,'selectiveUnits_dp_stim',file_postfix]);

load([rootpath2,'subAreaName']);
load([rootpath2,'areaInfo']);
load([firingRatePath,'spikeCount_go'])
load([selectivePath,'visualGradingSelective_allUnits'])

visualGrading_criteria=1;
rateThresh=1;%1HZ
cluster_comp='dp';%dp_interact,dp,stim
[areaNo,sessNo]=size(evidence.selective);
finalSelective=cell(areaNo,sessNo);
selVal=[-1,1];
selTitle={'contra','ipsi'};
selCount=zeros(2,areaNo);
unitCount_sess_ipsi=zeros(areaNo,39);
unitCount_sess_contra=zeros(areaNo,39);

postfix=cluster_comp;
if(visualGrading_criteria)
    postfix=[postfix,'_visualGrading'];
end

postfix=[postfix,file_postfix];

selective_areaName=cell(8,39);
for i=1 : areaNo
    if(i==7)
        continue;
    end
    for j=1 : sessNo
        if(isempty(evidence.selective{i,j}))
            continue;
        end
        
        finalSelective{i,j}=zeros(size(evidence.selective{i,j}));
        switch(cluster_comp)
            case 'stim'
                uidx=dpca.unitIdx{i}(dpca.clusters{i}==1 & dpca.sessIdx{i}==j);
            case 'dp'
                uidx=dpca.unitIdx{i}(dpca.clusters{i}==2 & dpca.sessIdx{i}==j);
            case 'dp_interact'
                uidx=dpca.unitIdx{i}((dpca.clusters{i}==2 | dpca.clusters{i}==3) & dpca.sessIdx{i}==j);
        end
        
        rateCriteria=avgFiringRate{i,j}'>=rateThresh;

        switch(cluster_comp)
            case {'dp','dp_interact'}
                for s=1 : 2
                    ev_selective=rateCriteria & evidence.selective{i,j}==selVal(s) & choice.selective{i,j}~=0;
                    if(visualGrading_criteria)
                        ev_selective=ev_selective & (~isnan(selectiveGrading{i,j}) & selectiveGrading{i,j}~=0); 
                    end
                    
                    finalSelective{i,j}(uidx)=finalSelective{i,j}(uidx)+selVal(s)*ev_selective(uidx);
                    selCount(s,i)=selCount(s,i)+sum(finalSelective{i,j}==selVal(s));
                end
                
            case 'stim'
                for s=1 : 2
                    ev_selective=rateCriteria & evidence.selective{i,j}==selVal(s) & contra.selective{i,j}~=0;
                    finalSelective{i,j}(uidx)=finalSelective{i,j}(uidx)+selVal(s)*ev_selective(uidx);
                    selCount(s,i)=selCount(s,i)+sum(finalSelective{i,j}==selVal(s));
                end
        end
        unitCount_sess_ipsi(i,j)=sum(finalSelective{i,j}==1);
        unitCount_sess_contra(i,j)=sum(finalSelective{i,j}==-1);
        
        idx_s=find(finalSelective{i,j}==1);
        if(~isempty(idx_s))
            selective_areaName{i,j}.contra=subArea{i,j}(idx_s);
        end
        
        idx_s=find(finalSelective{i,j}==-1);
        if(~isempty(idx_s))
            selective_areaName{i,j}.ipsi=subArea{i,j}(idx_s);
        end
    end
end

save([selectivePath,'finalSelective_clustering_',postfix],'finalSelective');
