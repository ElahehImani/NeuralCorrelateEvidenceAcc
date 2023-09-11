clc;clear;close all
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%%
rootpath='..\dataset\Mice\spike\result\';
rootpath2=[rootpath,'total_clusterArea_paper_results_kernel_0\'];
selectivePath=[rootpath2,'_auroc\selectiveUnits\'];
firingRatePath=[rootpath,'total_clusterArea_paper_results_kernel_0\_FiringRate\spikeCount_go'];

load([rootpath2,'areaInfo']);
load(firingRatePath);

all_units=0;
cluster_criteria=1;
cluster_comp='dp';%dp_interact,dp

out_file_postfix='';

if(cluster_criteria)
    out_file_postfix=[out_file_postfix,'_clustering_',cluster_comp];
end

out_file_postfix=[out_file_postfix,'_allUnits'];

load([selectivePath,'visualGraiding',out_file_postfix,'_Rate']);%load firing rates
load([selectivePath,'finalSelective',out_file_postfix]);%load firing rates

selVal=[-1,1];
selCount=zeros(2,length(spikeArea));
unitCount_sess=zeros(length(spikeArea),39);

selectiveGrading=cell(length(spikeArea),39);
evidenceLevel=[1,.75,.5,.25,0,-.25,-.5,-.75,-1];
colors=redbluecmap(11);
colors(6,:)=[];

visualFilterOpt=[];
visualFilterOpt.type='non-causal';%causal,non-causal
visualFilterOpt.kernel='gaussian';%half-gaussian, boxcar, exp, gaussian
visualFilterOpt.sigma=0.01;%in sec

outfile=[selectivePath,'visualGradingSelective_allUnits.mat'];
load([selectivePath,'visualGraiding_clustering_dp_allUnits_Rate'])


if(exist(outfile,'file'))
    load(outfile);
else
    area_pointer=1;
    contra_pointer=1;
    ipsi_pointer=1;
    for i=1 : length(spikeArea)
        for j=1 : 39
            if(~isempty(finalSelective{i,j}))
                selectiveGrading{i,j}=NaN(size(finalSelective{i,j}));
            end
        end
    end
end

idx=find(stim_times>=0);
zcross_stim=stim_times(idx(1));

idx=find(wheel_times>=0);
zcross_wheel=wheel_times(idx(1));
    
global vt;
while(area_pointer<=length(spikeArea))
    i=area_pointer;
    if(strcmp(spikeArea{i},'other'))
        area_pointer=area_pointer+1;
        contra_pointer=1;
        ipsi_pointer=1;
        continue;
    end
    
    %% contra units
    sessIdx=sessIdx_contra{i};
    unitIdx=unitIdx_contra{i};
    while(contra_pointer<length(sessIdx))
        u=contra_pointer;
        h=figure;
        maxAct=0;
        minAct=1000;
        sess=sessIdx(u);
        unit=unitIdx(u);
        for e=1 : 9
            subplot(1,2,1);hold on
            act=activityContra{e,i}(u,:);
            act=nonCausalSpikeFiltering(act,visualFilterOpt,stim_times);
            shadedErrorBar(stim_times,act,activityContra_sem{e,i}(u,:),'lineprops',{'color',colors(e,:),'LineWidth',1.5},'transparent',true);
            
            subplot(1,2,2);hold on
            act=activityContra_wheel{e,i}(u,:);
            act=nonCausalSpikeFiltering(act,visualFilterOpt,stim_times);
            shadedErrorBar(wheel_times,act,activityContra_wheel_sem{e,i}(u,:),'lineprops',{'color',colors(e,:),'LineWidth',1.5},'transparent',true);
            maxAct=max(maxAct,max(activityContra_wheel{e,i}(u,:)));
            minAct=min(minAct,min(activityContra_wheel{e,i}(u,:)));
        end
        
        subplot(1,2,1);axis square;
        plot([zcross_stim,zcross_stim],[minAct,maxAct],'k--')
        axis([stim_times(1),stim_times(end),minAct,maxAct]);
        title('contra')
        
        subplot(1,2,2);axis square;
        plot([zcross_wheel,zcross_wheel],[minAct,maxAct],'k--')
        axis([wheel_times(1),wheel_times(end),minAct,maxAct]);
        title([spikeArea{i},' fr=',num2str(avgFiringRate{i,sess}(unit),2)])
        
        vt = -1;
        h1= uicontrol('Style','pushbutton','String','good','callback',@setVt_c1,'position',[25 5 100 22]);
        h2= uicontrol('Style','pushbutton','String','bad','callback',@setVt_c2,'position',[125 5 100 22]);
        
        while(vt < 0)
            pause(1);
        end
        
        selectiveGrading{i,sess}(unit)=vt;
        save(outfile,'area_pointer','contra_pointer','ipsi_pointer','selectiveGrading');
        close(h)
        contra_pointer=contra_pointer+1;
    end
    
    %% ipsi units
    sessIdx=sessIdx_ipsi{i};
    unitIdx=unitIdx_ipsi{i};
    while(ipsi_pointer<length(sessIdx))
        u=ipsi_pointer;
        h=figure;
        maxAct=0;
        minAct=1000;
        sess=sessIdx(u);
        unit=unitIdx(u);
        for e=1 : 9
            subplot(1,2,1);hold on
            act=activityIpsi{e,i}(u,:);
            act=nonCausalSpikeFiltering(act,visualFilterOpt,stim_times);
            shadedErrorBar(stim_times,act,activityIpsi_sem{e,i}(u,:),'lineprops',{'color',colors(e,:),'LineWidth',1.5},'transparent',true);
            
            subplot(1,2,2);hold on
            act=activityIpsi_wheel{e,i}(u,:);
            act=nonCausalSpikeFiltering(act,visualFilterOpt,stim_times);
            shadedErrorBar(wheel_times,act,activityIpsi_wheel_sem{e,i}(u,:),'lineprops',{'color',colors(e,:),'LineWidth',1.5},'transparent',true);
            maxAct=max(maxAct,max(activityIpsi_wheel{e,i}(u,:)));
            minAct=min(minAct,min(activityIpsi_wheel{e,i}(u,:)));
        end
        
        subplot(1,2,1);axis square;
        plot([zcross_stim,zcross_stim],[minAct,maxAct],'k--')
        axis([stim_times(1),stim_times(end),minAct,maxAct]);
        title('ipsi')
        
        subplot(1,2,2);axis square;
        plot([zcross_wheel,zcross_wheel],[minAct,maxAct],'k--')
        axis([wheel_times(1),wheel_times(end),minAct,maxAct]);
        title([spikeArea{i},' fr=',num2str(avgFiringRate{i,sess}(unit),2)])
        
        vt = -1;
        h1= uicontrol('Style','pushbutton','String','good','callback',@setVt_c1,'position',[25 5 100 22]);
        h2= uicontrol('Style','pushbutton','String','bad','callback',@setVt_c2,'position',[125 5 100 22]);
        
        while(vt < 0)
            pause(1);
        end
        
        selectiveGrading{i,sess}(unit)=vt;
        save(outfile,'area_pointer','contra_pointer','ipsi_pointer','selectiveGrading');
        close(h)
        ipsi_pointer=ipsi_pointer+1;
    end
    
    area_pointer=area_pointer+1;
    contra_pointer=1;
    ipsi_pointer=1;
end

function setVt_c1(~,~)
    global vt; 
    vt = 1;
end

function setVt_c2(~,~)
    global vt; 
    vt = 0;
end
