%this script prepare data for the visual garding
clc;clear;close all
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']));
%%
subdir='total';
clusterName='clusterArea_paper';
regCluster=1;
if(regCluster)
    subdir=[subdir,'_',clusterName];
end

all_units=0;
rootpath='..\dataset\Mice\spike\result\';
ratepath=[rootpath,subdir,'_results_kernel_0\'];
baselinePath=[ratepath,'_baselineData\baselineData_kernel_0'];
behavPath=[rootpath,subdir,'\'];
choicePath=[rootpath,subdir,'_results_kernel_0\_auroc\'];
outpath=[choicePath,'selectiveUnits\'];

concatData=1;
cluster_criteria=1;
cluster_comp='dp';%dp_interact,dp

out_file_postfix='';

if(cluster_criteria)
    out_file_postfix=[out_file_postfix,'_clustering_',cluster_comp];
end

if(all_units)
    out_file_postfix=[out_file_postfix,'_allUnits'];
end

imgPath=[outpath,'_image',out_file_postfix,'\firingRate\'];
if(~exist(imgPath,'dir'))
    mkdir(imgPath);
end

load([ratepath,'areaInfo']);
load(baselinePath);
load([outpath,'finalSelective',out_file_postfix]);

dirinfo=dir(behavPath);
idx=strcmp({dirinfo.name},'.') | strcmp({dirinfo.name},'..');
dirinfo(idx)=[];

stim_boundary=[-.1,.3];
wheel_boundary=[-.3,.1];
sideSelective=[-1,1];
evidenceLevel=[1,.75,.5,.25,0,-.25,-.5,-.75,-1];
activityContra=cell(9,length(spikeArea));
activityIpsi=cell(9,length(spikeArea));
activityContra_wheel=cell(9,length(spikeArea));
activityIpsi_wheel=cell(9,length(spikeArea));

activityContra_sem=cell(9,length(spikeArea));
activityIpsi_sem=cell(9,length(spikeArea));
activityContra_wheel_sem=cell(9,length(spikeArea));
activityIpsi_wheel_sem=cell(9,length(spikeArea));

colors=redbluecmap(11);
colors(6,:)=[];

unitCount=zeros(2,length(spikeArea));
sessIdx_ipsi=cell(length(spikeArea),1);
unitIdx_ipsi=cell(length(spikeArea),1);
selectivityType_ipsi=cell(length(spikeArea),1);

sessIdx_contra=cell(length(spikeArea),1);
unitIdx_contra=cell(length(spikeArea),1);
selectivityType_contra=cell(length(spikeArea),1);

%%
for i=1 : 39
    load([behavPath,dirinfo(i).name,'\BasicInfo']);
    stimData=load([ratepath,dirinfo(i).name,'\stimSpikeRates_trlFixLen_1']);
    wheelData=load([ratepath,dirinfo(i).name,'\wheelSpikeRates_trlFixLen_1']);
    bound=stimData.stimBounday(1) : stimData.dt : stimData.stimBounday(2);
    idx1=find(bound>=stim_boundary(1));
    idx2=find(bound>=stim_boundary(2));
    sbound=idx1(1) : idx2(1);
    stim_times=bound(sbound);
    bound=wheelData.wheelBounday(1) : wheelData.dt : wheelData.wheelBounday(2);
    idx1=find(bound>=wheel_boundary(1));
    idx2=find(bound>=wheel_boundary(2));
    wbound=idx1(1) : idx2(1);
    wheel_times=bound(wbound);
    right=trialInfo.rightContrastLevel;
    left=trialInfo.leftContrastLevel;
    correct=trialInfo.correct;
    evidence=right-left;
    
    area=stimData.processedArea;
    cc=0;
    for j=1 : length(area)
        if(~isempty(find(strcmp(area{j},'invalid'))))
            continue;
        end
        cc=cc+1;
        aidx=find(strcmp(spikeArea,area{j}));
        if(isempty(aidx))
            continue;
        end
        for selective=1 : 2
            uidx=find(finalSelective{aidx,i}==sideSelective(selective));
            if(isempty(uidx))
                continue;
            end
            
            if(selective==1)
                sessIdx_ipsi{aidx}=[sessIdx_ipsi{aidx};i*ones(length(uidx),1)];
                unitIdx_ipsi{aidx}=[unitIdx_ipsi{aidx};uidx];
                selectivityType_ipsi{aidx}=[selectivityType_ipsi{aidx};sideSelective(selective)*ones(length(uidx),1)];
            else
                sessIdx_contra{aidx}=[sessIdx_contra{aidx};i*ones(length(uidx),1)];
                unitIdx_contra{aidx}=[unitIdx_contra{aidx};uidx];
                selectivityType_contra{aidx}=[selectivityType_contra{aidx};sideSelective(selective)*ones(length(uidx),1)];
            end
            
            unitCount(selective,aidx)=unitCount(selective,aidx)+length(uidx);
            baseMean=nanmean(baselineData{aidx,i}.avgRate,1);
            baseStd=nanstd(baselineData{aidx,i}.avgRate,[],1);
            for e=1 : 9
                trl=find(evidence==evidenceLevel(e) & correct);
                for u=1 : length(uidx)
                    act=NaN(1,length(sbound));
                    act_wheel=NaN(1,length(sbound));
                    for t=1 : length(trl)
                        act(t,:)=stimData.stimSpikeRate{trl(t),cc}(uidx(u),sbound);
                        act_wheel(t,:)=wheelData.wheelSpikeRate{trl(t),cc}(uidx(u),wbound);
                    end
                    
                    if(baseMean(uidx(u))~=0)
                        act=(act-baseMean(uidx(u)))./baseStd(uidx(u));
                        act_wheel=(act_wheel-baseMean(uidx(u)))./baseStd(uidx(u));
                        if(selective==1)
                            len=size(activityIpsi{e,aidx},1);
                            activityIpsi{e,aidx}(len+1,:)=nanmean(act,1);
                            activityIpsi_sem{e,aidx}(len+1,:)=nanstd(act,[],1)/sqrt(size(act,1));
                            len=size(activityIpsi_wheel{e,aidx},1);
                            activityIpsi_wheel{e,aidx}(len+1,:)=nanmean(act_wheel,1);
                            activityIpsi_wheel_sem{e,aidx}(len+1,:)=nanstd(act_wheel,[],1)/sqrt(size(act_wheel,1));
                        else
                            len=size(activityContra{e,aidx},1);
                            activityContra{e,aidx}(len+1,:)=nanmean(act,1);
                            activityContra_sem{e,aidx}(len+1,:)=nanstd(act,[],1)/sqrt(size(act,1));
                            len=size(activityContra_wheel{e,aidx},1);
                            activityContra_wheel{e,aidx}(len+1,:)=nanmean(act_wheel,1);
                            activityContra_wheel_sem{e,aidx}(len+1,:)=nanstd(act_wheel,[],1)/sqrt(size(act_wheel,1));
                        end
                    end
                end
            end
        end
    end
    disp(['session ',num2str(i)])
end

save([outpath,'visualGraiding',out_file_postfix,'_Rate'],'activityIpsi','activityIpsi_wheel',...
    'activityContra','activityContra_wheel','activityIpsi_sem','activityIpsi_wheel_sem','activityContra_sem','activityContra_wheel_sem',...
    'stim_times','wheel_times','unitCount','sessIdx_ipsi','unitIdx_ipsi','selectivityType_ipsi',...
    'sessIdx_contra','unitIdx_contra','selectivityType_contra');





