root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
clc;clear;close all
rng(1,'twister');
postfix='';
rootpath='..\dataset\Mice\result\total_clusterArea_paper_results_kernel_20_half_gaussian\';
matPath=[rootpath,'\_clustering\_dpca\'];
load([rootpath,'areaInfo'])

Reg={'frontal','hippocampus','striatum','visual','midbrain','thalamus','MOpSSp'};
myHierarchy={'hippocampus','thalamus','visual','striatum','frontal','MOpSSp','midbrain'};

colors2=[8,61,119;255,104,107;150,150,150]/255;

tau_grading_thresh=0;
condType=['stim_dp',postfix];%stim_dp,stim_choice

comp_idx=[1,2,4];

matPath=[matPath,condType,'\'];
load([matPath,'clustersInfo'])

imgpath=[matPath,'_image\'];
pdfpath=[matPath,'_pdf\'];

if(~exist(imgpath,'dir'))
    mkdir(imgpath);
end

if(~exist(pdfpath,'dir'))
    mkdir(pdfpath);
end
combinedR2=[];
combinedTau=[];
combinedPopR2=[];
for region=1 : length(myHierarchy)
    aidx=find(strcmp(myHierarchy{region},spikeArea));
    if(isempty(aidx))
        continue;
    end

    invalid=isnan(totalR2{aidx}(:,1));
    R2=totalR2{aidx}(~invalid,comp_idx);

    combinedR2=[combinedR2;R2];
    unitCount=sum(~invalid);

    %% clustering
    classes=clusters{aidx};
    classes=classes(~invalid);
    class_idx=[];
    [~,max_1]=max(R2(:,1));
    class_idx(1)=classes(max_1);
    [~,max_2]=max(R2(:,2));
    class_idx(2)=classes(max_2);
    class_idx(3)=setdiff(1:classNo,class_idx);

    for c=1 : length(class_idx)
        cidx=find(classes==class_idx(c));
        figure(1);subplot(2,4,region);hold on;
        scatter(R2(cidx,1),R2(cidx,2),20,'markerFaceColor',colors2(c,:),'markerEdgeColor','None');
        alpha 0.3
    end

    figure(1);subplot(2,4,region);hold on;
    xlabel([margNames{comp_idx(1)},' R^2'])
    ylabel([margNames{comp_idx(2)},' R^2'])
    axis square;title([spikeArea{aidx}]);axis([-5,70,-5,70])

    figure(2);subplot(2,4,region);hold on
    bar(1,totalPopR2{aidx}(1),'faceColor',colors2(1,:))
    bar(2,totalPopR2{aidx}(2),'faceColor',colors2(2,:))
    bar(3,totalPopR2{aidx}(4),'faceColor',colors2(3,:))
    xticks(1:3);xticklabels({'stimulus','decision','interaction'});xtickangle(45)
    axis square;xlim([0,4]);ylim([0,30]);ylabel('population R^2')
    combinedPopR2=[combinedPopR2;[totalPopR2{aidx}([1,2,4])]];
    title(spikeArea{aidx})
end

h=figure(1);subplot(2,4,8);hold on;
xlabel([margNames{comp_idx(1)},' R^2'])
ylabel([margNames{comp_idx(2)},' R^2'])
axis square;title('total area');axis([-5,70,-5,70])
set(h,'Position',[300,300,800,400])
saveas(h,[imgpath,'clustering.png'])
saveas(h,[pdfpath,'clustering.pdf'])


h=figure(2);
set(h,'Position',[300,300,800,400])
saveas(h,[imgpath,'population_var.png'])
saveas(h,[pdfpath,'population_var.pdf'])

h=figure(3);
ba=bar(combinedPopR2,'stacked', 'FaceColor','flat');
ba(1).CData = colors2(1,:);
ba(2).CData = colors2(2,:);
ba(3).CData = colors2(3,:);
xticks(1:7)
xticklabels(myHierarchy)
xtickangle(45)
legend('stimulus','decision','interaction','location','best')
xlim([0,8]);ylim([0,60])
ylabel('population R^2');axis square
saveas(h,[imgpath,'population_var_stacked.png'])
saveas(h,[pdfpath,'population_var_stacked.pdf'])
