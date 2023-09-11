clc;clear;close all
rng(1,'twister');
% computing PSTHs
clear
all_units=0;
rootpath='..\dataset\Mice\result\total_clusterArea_paper_results_kernel_0\_clustering\';

postfix='';
if(all_units)
    postfix=[postfix,'_allUnits'];
end

matPath=[rootpath,'_dpca\'];

load([rootpath,'clusteringFeature',postfix])

%as described in the demo this is the 4D matrix that you want
% N is the number of neurons
% S is the number of stimuli conditions (F1 frequencies in Romo's task)
% D is the number of decisions (D=2)
% T is the number of time-points (note that all the trials should have the


classNo=3;
Reg={'frontal','hippocampus','striatum','visual','midbrain','thalamus','MOpSSp'};
colors=[13,161,75;96,55,19;127,62,152;239,62,54;236,0,140;247,148,29;33,170,225]/255;
% colors2=[0,151,138;255,10,84;250,167,47;0,255,0]/255;
colors2=[8,61,119;255,104,107;150,150,150]/255;

combinedParams = {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}};

condType='stim_dp';%stim_dp,stim_choice

if(strcmp(condType,'stim_choice'))
    comp_idx=[1,2];
    d_list{1,1} = [90 91]; % R=0 right left
    d_list{1,2} = [95 96]; % R=25 right left
    d_list{1,3} = [100 101]; % R=50 right left
    d_list{1,4} = [105 106]; % R=100 right left
    
    margNames = {'Stimulus', 'Decision', 'Condition-independent', 'S/D Interaction'};
else
    if(all_units)
        comp_idx=[1,2];
    else
        comp_idx=[1,2,4];
    end
    margNames = {'Stimulus', 'Detect', 'Condition-independent', 'S/D Interaction'};
 
    d_list{1,1} = [88 89]; % R=0 Go NoGo
    d_list{1,2} = [93 94]; % R=25 Go NoGo
    d_list{1,3} = [98 99]; % R=50 Go NoGo
    d_list{1,4} = [103 104]; % R=100 Go NoGo
end

matPath=[matPath,condType,postfix,'\'];
imgpath=[matPath,'_image\'];
pdfpath=[matPath,'_pdf\'];

if(~exist(imgpath,'dir'))
    mkdir(imgpath);
end

if(~exist(pdfpath,'dir'))
    mkdir(pdfpath);
end

if(~exist(matPath,'dir'))
    mkdir(matPath);
end

compNo=20;
% Define parameter grouping

% *** Don't change this if you don't know what you are doing! ***
% firingRates array has [N S D T E] size; herewe ignore the 1st dimension
% (neurons), i.e. we have the following parameters:
%    1 - stimulus
%    2 - decision
%    3 - time
% There are three pairwise interactions:
%    [1 3] - stimulus/time interaction
%    [2 3] - decision/time interaction
%    [1 2] - stimulus/decision interaction
% And one three-way interaction:
%    [1 2 3] - rest
% As explained in the eLife paper, we group stimulus with stimulus/time interaction etc.:


% For two parameters (e.g. stimulus and time, but no decision), we would have
% firingRates array of [N S T E] size (one dimension less, and only the following
% possible marginalizations:
%    1 - stimulus
%    2 - time
%    [1 2] - stimulus/time interaction
% They could be grouped as follows:
%    combinedParams = {{1, [1 2]}, {2}};

% for later stages only look at tau
% load([rootpath,'R2'])
totalR2=cell(1,length(spikeArea));
totalPopR2=cell(1,length(spikeArea));
dpcaProjection=cell(1,length(spikeArea));
dpcaParams=[];
combinedR2=[];

% for region=1 : length(Reg)
for aidx=1 : length(spikeArea)
    aidx2=find(strcmp(spikeArea{aidx},Reg));
    if(isempty(aidx2))
        continue;
    end

    R2=NaN(length(unitIdx{aidx}),size(d_list,1),length(margNames));
    Z=NaN(size(d_list,1),compNo,length(times)*size(d_list,2)*length(d_list{1,1}));
    pop_R2=NaN(size(d_list,1),length(margNames));
    sw=1;
    for f=1 : size(d_list,1)
        all_features = [];
        for j=1 : size(d_list,2)
            all_features=[all_features,d_list{f,j}];
        end
        
        %generate matrix from neurons that don't have NaN values for ANY of the
        %features of interest
        hasnan=[];
        for i = 1:size(features{aidx},1)
            for i2 = 1:numel(all_features)
                hasnan(i,i2) = any(isnan(features{aidx,all_features(i2)}(i,:)));
            end
        end
        
        neuron_idx = ~[sum(hasnan')>0];

        neuron_idx=neuron_idx';
        
        features2 =cell(8,86);
        for i = 1:numel(all_features)
            features2{aidx,all_features(i)} = features{aidx,all_features(i)}(neuron_idx,:);
        end
        
        %Generate 4d matrix that can be used by DPCA
        B=[];
        unitIdx_list = unitIdx{aidx}(neuron_idx);
        for n = 1:length(unitIdx_list) % x neuron
            for n2 = 1:size(d_list,2) % neuron x stim
                for n3 = 1:numel(d_list{f,n2}) % neuron x stim x choice
                    B(n,n2,n3,:) = features2{aidx,d_list{f,n2}(n3)}(n,:); % neuron x stim x choice x time
                end
            end
        end

        if(isempty(B))
            sw=0;
            continue;
        end
        
        if(size(B,1)<20)
            sw=0;
            dpcaParams(aidx).W=[];
            dpcaParams(aidx).V=[];
            dpcaParams(aidx).times=[];
            dpcaParams(aidx).whichMarg=[];
            continue;
        end

        %this is the variable name that DPCA takes in the demo so I just used it
        firingRatesAverage = B;

        % Step 3: dPCA without regularization and ignoring noise covariance
        tic
        [W,V,whichMarg] = dpca(firingRatesAverage, compNo, ...
            'combinedParams', combinedParams);
        toc
        
        dpcaParams(aidx).W=W;
        dpcaParams(aidx).times=times;
        dpcaParams(aidx).V=V;
        dpcaParams(aidx).whichMarg=whichMarg;
        [R2(neuron_idx,f,:),Z(f,:,:),pop_R2(f,:)]=unitExplainedVar(firingRatesAverage,W,V,margNames,whichMarg);
    end

    if(aidx==striatum_idx)
        R2(snr_idx,:,:)=[];
    end

    dataLen=size(firingRatesAverage);
    Z=squeeze(nanmean(Z,1));
    Z=reshape(Z,[size(Z,1),dataLen(2:end)]);
    totalR2{aidx}=squeeze(nanmean(R2,2));
    totalPopR2{aidx}=nanmean(pop_R2,1);
    dpcaProjection{aidx}=Z;
    invalid=isnan(totalR2{aidx}(:,1));
    totalR2{aidx}(invalid,:)=NaN;
    
    R2=totalR2{aidx}(~invalid,comp_idx);
    if(~sw)
        continue;
    end


    %% clustering
    clusters{aidx}=NaN(size(invalid));
    [centers,U] = fcm(R2,classNo);
    [~,classes]=max(U,[],1);
    
    if(strcmp(condType,'stim_dp'))
        if(~all_units)
            class_idx=[];
            [~,max_1]=max(R2(:,1));
            class_idx(1)=classes(max_1);
            [~,max_2]=max(R2(:,2));
            class_idx(2)=classes(max_2);
            class_idx(3)=setdiff(1:classNo,class_idx);
        else
            class_idx=[];
            [~,I]=max(R2(:,2));
            class_idx(2)=classes(I(1));
            Dist=sqrt(R2(:,1).^2+R2(:,2).^2);
            [~,I]=min(Dist);
            class_idx(3)=classes(I(1));
            class_idx(1)=setdiff(1:classNo,class_idx);
        end
    else
        class_idx=[];
        [~,I]=max(R2(:,1));
        class_idx(1)=classes(I(1));
        Dist=sqrt(R2(:,1).^2+R2(:,2).^2);
        [~,I]=min(Dist);
        class_idx(3)=classes(I(1));
        class_idx(2)=setdiff(1:classNo,class_idx);
    end
    
    clusterCenters{aidx}=centers;
    
    class_tau=[];
    cluster_tmp=NaN(size(R2,1),1);
    
    class_unit_num=zeros(1,classNo);
    for c=1 : classNo
        class_unit_num(c)=sum(classes==class_idx(c));
    end
    
    [~,class_order]=sort(class_unit_num,'descend');
    for cc=1 : length(class_idx)
        c=class_order(cc);%I just use this for hisgoram plotting (I want to show the bigger hist first)
        cidx=find(classes==class_idx(c));
        cluster_tmp(cidx)=c;
    end
    
    clusters{aidx}(~invalid)=cluster_tmp;
end

save([matPath,'clustersInfo'],'times','clusters','clusterCenters','totalR2','totalPopR2','dpcaProjection','margNames','classNo','dpcaParams','SubArea','sessIdx','unitIdx');

