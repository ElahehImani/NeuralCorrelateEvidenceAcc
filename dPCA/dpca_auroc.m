clc;clear;close all
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%%
rootpath='..\dataset\Mice\result\total_clusterArea_paper_results_kernel_20_half_gaussian\';
rocPath=[rootpath,'_auroc\'];
dpcaPath=[rootpath,'_clustering\_dpca\stim_dp_allUnits\'];

load([rootpath,'areaInfo']);
load([dpcaPath,'clustersInfo'])
stim=load([rocPath,'temporal_contra_stim_allUnits']);
dp=load([rocPath,'temporal_dp_stim_allUnits']);
load([rootpath,'_clustering\clusteringFeature_allUnits']);

imgPath=[dpcaPath,'_image\'];
pdfpath=[dpcaPath,'_pdf\'];

Reg={'frontal','hippocampus','striatum','visual','midbrain','thalamus','MOpSSp'};
myHierarchy={'hippocampus','thalamus','visual','striatum','frontal','MOpSSp','midbrain'};
colors2=[13,161,75;96,55,19;127,62,152;239,62,54;236,0,140;247,148,29;33,170,225]/255;
colors=[240,1,77;0,145,110;150,150,150]/255;

visualFilterOpt=[];
visualFilterOpt.type='non-causal';%causal,non-causal
visualFilterOpt.kernel='gaussian';%half-gaussian, boxcar, exp, gaussian
visualFilterOpt.sigma=0.02;%in sec
times=stim.times;
zcross=find(times>=0);
zcross=times(zcross(1));

stim_ylimt=[0.45,0.7];
dp_ylimt=[0.45,0.75];

totalCount=zeros(1,8);
for i=1 : length(spikeArea)
    aidx=find(strcmp(spikeArea{i},myHierarchy));
    if(isempty(aidx))
        continue;
    end
    
    h=figure(1);
    subplot(2,4,aidx);hold on;
    plot([times(1),times(end)],[0.5,0.5],'k--')
    plot([zcross,zcross],stim_ylimt,'k--')
    
    h=figure(2);
    subplot(2,4,aidx);hold on;
    plot([times(1),times(end)],[0.5,0.5],'k--')
    plot([zcross,zcross],dp_ylimt,'k--')
    unit_num=zeros(1,3);
    for j=1 : 3
        idx=find(clusters{i}==j);
        unit_num(j)=length(idx);
        uidx=unitIdx{i}(idx);
        sess=sessIdx{i}(idx);
        usess=unique(sess);
        roc_stim=[];
        roc_dp=[];
        for s=1 : length(usess)
            idx=find(sess==usess(s));
            roc_stim=[roc_stim;stim.decodingRes{i,usess(s)}(uidx(idx),:)];
            roc_dp=[roc_dp;dp.decodingRes{i,usess(s)}(uidx(idx),:)];
        end
        
        figure(1);subplot(2,4,aidx);
        meanAcc=nanmean(roc_stim,1);
        stdAcc=1.96*nanstd(roc_stim,1)/sqrt(size(roc_stim,1));
        meanAcc=nonCausalSpikeFiltering(meanAcc,visualFilterOpt,times);
        stdAcc=nonCausalSpikeFiltering(stdAcc,visualFilterOpt,times);
    
        shadedErrorBar(times,meanAcc,stdAcc,'lineprops',{'color',colors(j,:),'LineWidth',1},'transparent',true);
        

        figure(2);subplot(2,4,aidx);
        meanAcc=nanmean(roc_dp,1);
        stdAcc=1.96*nanstd(roc_dp,1)/sqrt(size(roc_dp,1));
        meanAcc=nonCausalSpikeFiltering(meanAcc,visualFilterOpt,times);
        stdAcc=nonCausalSpikeFiltering(stdAcc,visualFilterOpt,times);
    
        shadedErrorBar(times,meanAcc,stdAcc,'lineprops',{'color',colors(j,:),'LineWidth',1},'transparent',true);
    end
    totalCount(i)=sum(unit_num);
    
    figure(1);subplot(2,4,aidx);axis square
    title(spikeArea{i})
    axis([times(1),times(end),stim_ylimt])
    
    figure(2);subplot(2,4,aidx);axis square
    title(spikeArea{i})
    axis([times(1),times(end),dp_ylimt])

    figure(3);subplot(2,4,aidx);
    ax=gca();
    pie(unit_num/sum(unit_num));
    ax.Colormap=colors;
    title(spikeArea{i})
end

h=figure(1);
set(h,'Position',[100,100,800,500])
saveas(h,[imgPath,'stim.png'])
saveas(h,[pdfpath,'stim.pdf'])

h=figure(2);
set(h,'Position',[100,100,800,500])
saveas(h,[imgPath,'DP.png'])
saveas(h,[pdfpath,'DP.pdf'])

h=figure(3);
set(h,'Position',[100,100,700,400])
saveas(h,[imgPath,'pie.png'])
saveas(h,[pdfpath,'pie.pdf'])

h=figure(4);hold on;
for i=1 : length(myHierarchy)
    aidx=find(strcmp(myHierarchy{i},spikeArea));
    aidx2=find(strcmp(myHierarchy{i},Reg));
    
    bar(i,totalCount(aidx),'FaceColor',colors2(aidx2,:));
end
xticks(1:7);xticklabels(myHierarchy);xtickangle(45);
ylabel('Cell Number');axis square
set(h,'Position',[100,100,250,250])
saveas(h,[imgPath,'totalCount.png'])
saveas(h,[pdfpath,'totalCount.pdf'])

