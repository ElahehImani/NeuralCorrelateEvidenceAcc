root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
clc;clear;close all
rootpath='..\dataset\Mice\result\total_clusterArea_paper_results_kernel_20_half_gaussian\';
dpcaPath=[rootpath,'_clustering\_dpca\stim_dp_allUnits\'];
load([dpcaPath,'clustersInfo']);
load([rootpath,'areaInfo']);
imgPath=[dpcaPath,'_image_projection\'];
if(~exist(imgPath,'dir'))
    mkdir(imgPath);
end

Reg={'frontal','hippocampus','striatum','visual','midbrain','thalamus','MOpSSp'};

visualFilterOpt=[];
visualFilterOpt.type='non-causal';%causal,non-causal
visualFilterOpt.kernel='gaussian';%half-gaussian, boxcar, exp, gaussian
visualFilterOpt.sigma=0.02;%in sec


idx1=find(times>=0);
zcross=times(idx1(1));

ylim_stim=[-80,80];
ylim_decision=[-30,30];
ylim_interaction=[-60,60];

stimulus_colors={[0 0.4470 0.7410],[0.8500 0.3250 0.0980]};
choice_colors={'k','k'};
choice_lineStyle={'-','--'};
stim_comp=[1,4];
choice_comp=[1,2];
for i=1 : length(spikeArea)
    aidx=find(strcmp(spikeArea{i},Reg));
    if(isempty(aidx))
        continue;
    end
    
    stim_idx=find(dpcaParams(i).whichMarg==1);
    stim_idx=stim_idx(1);
    choice_idx=find(dpcaParams(i).whichMarg==2);
    choice_idx=choice_idx(1);
    interaction_idx=find(dpcaParams(i).whichMarg==4);
    interaction_idx=interaction_idx(1);
    
    h=figure(1);
    subplot(7,3,(aidx-1)*3+1);hold on
    plot(times,zeros(1,length(times)),'k--');plot([])
    plot([zcross,zcross],ylim_stim,'k--');

    for s=1 : 2
        for d=1 : 2
            Y=squeeze(nanmean(dpcaProjection{i}(stim_idx,stim_comp(s),choice_comp(d),:),3));
            Y=nonCausalSpikeFiltering(Y,visualFilterOpt,times);
            shadedErrorBar(times,Y,zeros(1,length(times)),'lineprops',{choice_lineStyle{d},'color',stimulus_colors{s},'LineWidth',1},'transparent',true,'patchSaturation',0.5);
        end
    end
    
    axis square;xlim([times(1),times(end)]);ylim(ylim_stim)
    ylabel('stim-proj');title(spikeArea{i})
    
    % -------------------------------------
    subplot(7,3,(aidx-1)*3+2);hold on
    plot(times,zeros(1,length(times)),'k--');
    plot([zcross,zcross],ylim_decision,'k--');
    for s=1 : 2
        for d=1 : 2
            Y=squeeze(nanmean(dpcaProjection{i}(choice_idx,stim_comp(s),choice_comp(d),:),2));
            Y=nonCausalSpikeFiltering(Y,visualFilterOpt,times);
            shadedErrorBar(times,Y,zeros(1,length(times)),'lineprops',{choice_lineStyle{d},'color',stimulus_colors{s},'LineWidth',1},'transparent',true,'patchSaturation',0.5);
        end
    end
    
    axis square;xlim([times(1),times(end)]);ylim(ylim_decision)
    ylabel('decision-proj');title(spikeArea{i})

    % -------------------------------------
    subplot(7,3,(aidx-1)*3+3);hold on
    plot(times,zeros(1,length(times)),'k--');
    plot([zcross,zcross],ylim_interaction,'k--');
    for s=1 : 2
        for d=1 : 2
            Y=squeeze(nanmean(dpcaProjection{i}(interaction_idx,stim_comp(s),choice_comp(d),:),2));
            Y=nonCausalSpikeFiltering(Y,visualFilterOpt,times);
            shadedErrorBar(times,Y,zeros(1,length(times)),'lineprops',{choice_lineStyle{d},'color',stimulus_colors{s},'LineWidth',1},'transparent',true,'patchSaturation',0.5);
        end
    end
    
    axis square;xlim([times(1),times(end)]);ylim(ylim_interaction)
    ylabel('interaction-proj');title(spikeArea{i})

end

set(h,'Position',[100,100,600,1500])
saveas(h,[imgPath,'total.png'])
saveas(h,[imgPath,'total.pdf'])