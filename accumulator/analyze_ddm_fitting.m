clc;clear;close all
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
%% ------------------------------------------------------------------------
rootpath='..\dataset\Mice\result\Accumulator\spikes_subsample_postWheel_bestfiles\';
data_path='..\dataset\Mice\result\total_clusterArea_paper_results_kernel_20_half_gaussian\';
load([data_path,'areaInfo']);
lagNo=9;
lag=0.025;
bins=0:lag:0.25;
imgPath=[rootpath,'_images\'];
if(~exist(imgPath,'dir'))
    mkdir(imgPath);
end
latencyThresh=0.15;
method_total={'','single_collaps','','race_collaps','','compete_collaps'};
method={'single','race','compete'};
param_number=[23,25,39,41,87,89];%'single','race','compete'
colors=[13,161,75;127,62,152;239,62,54;236,0,140;33,170,225;247,148,29]/255;
regions={'frontal','striatum','visual','midbrain','MOpSSp','thalamus','midbrain_striatum'};
myHierarchy={'thalamus','visual','striatum','MOpSSp','frontal','midbrain'};

sel_color_single=[255,189,53]/255;
sel_color_race=[255,0,117]/255;
sel_color_compete=[46,176,134]/255;

R2_totalBoundary=cell(length(method_total),length(regions));
R2=cell(length(method),length(regions));
model_autocorr_totalBoundary=cell(length(method_total),length(regions));
model_autocorr=cell(length(method),length(regions));
model_tau_totalBoundary=cell(length(method_total),length(regions));
model_tau=cell(length(method),length(regions));
R2_units=cell(length(method_total),length(regions));
R2_units_bestModel_combined=cell(1,length(regions));
R2_units_bestModel=cell(1,length(regions));
LL_totalBoundary=cell(length(method_total),length(regions));
LL=cell(length(method),length(regions));
AIC_totalBoundary=cell(length(method_total),length(regions));
AIC=cell(length(method),length(regions));
A_matrix_totalBoundary=cell(length(method_total),length(regions));
A_matrix=cell(length(method_total),length(regions));
A_init_totalBoundary=cell(length(method_total),length(regions));
A_init=cell(length(method_total),length(regions));
A_sign_totalBoundary=cell(length(method_total),length(regions));
A_sign=cell(length(method_total),length(regions));
unit_sides=cell(1,length(regions));
unit_selectivity=cell(1,length(regions));
unitArea=cell(1,length(regions));
fileNames=cell(1,length(regions));

%% reading data
for i=1 : length(regions)
    regPath=[rootpath,regions{i},'\'];
    dirinfo=dir(fullfile(regPath,'*.mat'));% read main data
    r2_units=cell(1,length(method));
    unit_Idx=[];sess_Idx=[];Tau_units_total=[];WaveAmp_units_total=[];
    fc=0;
    for f=1 : length(dirinfo)
        unit_data=load([regPath,dirinfo(f).name]);
        if(unit_data.latency_ms>latencyThresh)
            continue;
        end

        fc=fc+1;
        fileNames{i}{fc}=dirinfo(f).name;

        %% ------------------------------------------------------------
        unit_selectivity{i}=[unit_selectivity{i};sum(unit_data.side==-1)>0 & sum(unit_data.side==1)>0];

        unit_sides{i}=[unit_sides{i};unit_data.side];

        for u=1 : length(unit_data.side)
            unitArea{i}(fc,u)=unit_data.area{u};
        end

        aidx=find(strcmp(spikeArea,regions{i}));

        for m=1 : length(method_total)
            name_parts=split(dirinfo(f).name,'.mat');
            fittingPath=[regPath,'_results_',method_total{m},'\_BestLogLike\'];
            tauPath=[regPath,'_results_',method_total{m},'\_tau_lag25\'];
            filename_r2=[name_parts{1},'_best_EV.mat'];
            filename_tau=[name_parts{1},'_tau.mat'];

            %% load tau files ----------------------------
            if(exist([tauPath,filename_tau],'file'))
                mdl_tau=load([tauPath,filename_tau]);
                model_tau_totalBoundary{m,i}(fc,1)=mdl_tau.tau;
                model_autocorr_totalBoundary{m,i}(fc,:)=mdl_tau.autocorr;
            else
                model_tau_totalBoundary{m,i}(fc,1)=NaN;
                model_autocorr_totalBoundary{m,i}(fc,:)=NaN(1,lagNo);
            end

            %% load R2 and loglike files -----------------
            if(~exist([fittingPath,filename_r2],'file'))
                R2_totalBoundary{m,i}(fc,:)=NaN(1,100);
                r2_units{m}(fc,:)=NaN(1,4);
                LL_totalBoundary{m,i}(fc,1)=NaN;
                AIC_totalBoundary{m,i}(fc,1)=NaN;
                if(m==1 || m==2)
                    A_matrix_totalBoundary{m,i}(fc,:,:)=NaN;
                else
                    A_matrix_totalBoundary{m,i}(fc,:,:)=NaN(2,2);
                end

                A_init_totalBoundary{m,i}(fc,:)=NaN;
                A_sign_totalBoundary{m,i}(fc)=NaN;
                continue;
            end

            % load results
            R2_data=load([fittingPath,filename_r2]);
            R2_totalBoundary{m,i}(fc,:)=R2_data.totalR2_pop(:,1:100);
            idx=isinf(R2_totalBoundary{m,i}(fc,:)) | R2_totalBoundary{m,i}(fc,:)<-2;
            R2_totalBoundary{m,i}(fc,idx)=NaN;

            r2_units{m}(fc,:)=nanmean(R2_data.totalR2_units,2);
            LL_totalBoundary{m,i}(fc,:)=R2_data.logLike;
            AIC_totalBoundary{m,i}(fc,1)=2*(param_number(m))-2*R2_data.logLike;%default number of params is 2
            if(R2_data.logLike==0)
                AIC_totalBoundary{m,i}(fc,1)=NaN;
            end
            if(strcmp(method_total{m},'single_collaps'))
                if(ndims(R2_data.trained_A)==1)
                    A_matrix_totalBoundary{m,i}(fc,:,:)=R2_data.trained_A(1);
                else
                    A_matrix_totalBoundary{m,i}(fc,:,:)=squeeze(R2_data.trained_A(1,:,:));
                end
            end

            if(strcmp(method_total{m},'race_collaps') || strcmp(method_total{m},'compete_collaps'))
                if(ndims(R2_data.trained_A)==2)
                    A_matrix_totalBoundary{m,i}(fc,:,:)=R2_data.trained_A(:,:);
                else
                    A_matrix_totalBoundary{m,i}(fc,:,:)=squeeze(R2_data.trained_A(1,:,:));
                end
            end

            A_sign_totalBoundary{m,i}(fc)=1;
            tmp=squeeze(R2_data.trained_A(1,:,:));
            tmp=tmp(:);
            if(sum(tmp<0)>0)
                A_sign_totalBoundary{m,i}(fc)=-1;
            end
        end
    end

    sess_Idx=repmat(sess_Idx,1,size(unit_Idx,2));

    for m=1 : length(method)
        a=(m-1)*2+1;
        b=(m-1)*2+2;
        d_aic=[AIC_totalBoundary{a,i},AIC_totalBoundary{b,i}];
        d_aic(isinf(d_aic))=NaN;
        [AIC{m,i},boundaryType{m,i}]=nanmin(d_aic,[],2);
        for u=1 : size(d_aic,1)
            if(boundaryType{m,i}==1)
                R2{m,i}(u,1)=nanmean(R2_totalBoundary{a,i}(u,:),2);
                A_matrix{m,i}(u,:,:)=A_matrix_totalBoundary{a,i}(u,:,:);
                A_sign{m,i}(u)=A_sign_totalBoundary{a,i}(u);
                LL{m,i}(u,:)=LL_totalBoundary{a,i}(u,:);
                model_tau{m,i}(u,1)=model_tau_totalBoundary{a,i}(u);
                model_autocorr{m,i}(u,:)=model_autocorr_totalBoundary{a,i}(u,:);
            else
                R2{m,i}(u,1)=nanmean(R2_totalBoundary{b,i}(u,:),2);
                A_matrix{m,i}(u,:,:)=A_matrix_totalBoundary{b,i}(u,:,:);
                A_sign{m,i}(u)=A_sign_totalBoundary{b,i}(u);
                LL{m,i}(u,:)=LL_totalBoundary{b,i}(u,:);
                model_tau{m,i}(u,1)=model_tau_totalBoundary{b,i}(u);
                model_autocorr{m,i}(u,:)=model_autocorr_totalBoundary{b,i}(u,:);
            end
        end
    end
end

%% number of units
ucount=zeros(2,length(myHierarchy));
h=figure;hold on;
for i=1 : length(myHierarchy)
    aidx=find(strcmp(myHierarchy{i},regions));
    ucount(1,i)=sum(unit_selectivity{aidx}==1);
    ucount(2,i)=sum(unit_selectivity{aidx}==0);
end

h=figure;hold on;
bar(ucount','stacked','BarWidth',0.7)
xticks(1:6)
xticklabels(myHierarchy);xtickangle(45)
axis([0,7,0,180])
axis square
saveas(h,[imgPath,'subsampleNum.pdf']);
close(h)

%% best model for each pop
totalR=cell(1,length(regions));
total_modelTau=cell(1,length(regions));
total_modelAutocorr=cell(1,length(regions));
best_modelTau=cell(1,length(regions));
best_modelAutocorr=cell(1,length(regions));
unit_preference_index_unilateral=cell(1,length(regions));
unit_preference_index=cell(3,length(regions));

aic_lim=1000;
unit_per_model=zeros(length(regions),length(method));
unit_per_model_unilateral=zeros(length(regions),1);
total_signle_pref=[];
total_race_pref=[];
total_compete_pref=[];
unstable_cnt=0;
for i=1 : length(myHierarchy)
    R2_units_bestModel_combined{i}=cell(length(unique_unitIdx{i}),1);
    R2_units_bestModel{i}=NaN(length(unique_unitIdx{i}),1);
    for j=1 : length(method)
        totalR{i}(:,j)=R2{j,i};
    end

    for j=1 : length(method)
        totalAIC{i}(:,j)=AIC{j,i};
        total_modelTau{i}(:,j)=model_tau{j,i};
        total_modelAutocorr{i}(:,j,:)=model_autocorr{j,i};
    end

    single_race=totalAIC{i}(:,1)-totalAIC{i}(:,2);
    single_compete=totalAIC{i}(:,1)-totalAIC{i}(:,3);
    race_compete=totalAIC{i}(:,2)-totalAIC{i}(:,3);

    %% AIC difference
    aic_diff=totalAIC{i}-nanmin(totalAIC{i},[],2);
    [r,c]=find(aic_diff>0 & aic_diff<=10);
    [~,idx]=nanmin(aic_diff,[],2);
    invalid=zeros(size(aic_diff,1),1);
    invalid(r)=1;
    single_models=(invalid==0 & idx==1 & unit_selectivity{i}>0) & totalR{i}(:,1)>0;
    race_models=(invalid==0 & idx==2 & unit_selectivity{i}>0) & totalR{i}(:,2)>0;
    compete_models=(invalid==0 & idx==3 & unit_selectivity{i}>0) & totalR{i}(:,3)>0;

    %%
    single_models_unilateral=(unit_selectivity{i}==0) & totalR{i}(:,1)>0;
    single_models_unilateral(unstable_idx{1,i})=0;

    bestModel=zeros(size(single_models));
    bestModel(find(single_models))=1;
    bestModel(find(race_models))=2;
    bestModel(find(compete_models))=3;
    bestModel(find(single_models_unilateral))=4;

    best_modelTau{i}=NaN(length(single_models),1);
    best_modelTau{i}(find(single_models))=total_modelTau{i}(find(single_models),1);
    best_modelTau{i}(find(race_models))=total_modelTau{i}(find(race_models),2);
    best_modelTau{i}(find(compete_models))=total_modelTau{i}(find(compete_models),3);
    best_modelTau{i}(find(single_models_unilateral))=total_modelTau{i}(find(single_models_unilateral),1);

    best_modelAutocorr{i}=NaN(length(single_models),size(model_autocorr{1,1},2));
    best_modelAutocorr{i}(find(single_models),:)=total_modelAutocorr{i}(find(single_models),1,:);
    best_modelAutocorr{i}(find(race_models),:)=total_modelAutocorr{i}(find(race_models),2,:);
    best_modelAutocorr{i}(find(compete_models),:)=total_modelAutocorr{i}(find(compete_models),3,:);
    best_modelAutocorr{i}(find(single_models_unilateral),:)=total_modelAutocorr{i}(find(single_models_unilateral),1,:);

    total_valid_num=sum(~isnan(bestModel));
    for j=1 : length(method)
        unit_per_model(i,j)=sum(bestModel==j);
    end
    unit_per_model(i,:)=unit_per_model(i,:)/total_valid_num;

    unit_per_model_unilateral(i,1)=nansum(single_models_unilateral);
    unit_per_model_unilateral(i,:)=unit_per_model_unilateral(i,:)/total_valid_num;
    unit_preference_index_unilateral{1,i}=find(single_models_unilateral);

    for j=1 : length(method)
        idx=find(bestModel==j);
        unit_preference_index{j,i}=idx;

        % prefered model - unique units R2
        for k=1 : length(unique_unitIdx{i})
            [Row,Col]=find(unitIdx{i}==unique_unitIdx{i}(k) & sessIdx{i}==unique_sessIdx{i}(k));

            for u=1 : length(Row)
                idx=find(Row(u)==unit_preference_index{j,i});
                if(~isempty(idx))
                    R2_units_bestModel_combined{i}{k}=[R2_units_bestModel_combined{i}{k};totalR{i}(Row(u),j)];
                end
            end
        end
    end

    for k=1 : length(unique_unitIdx{i})
        if(~isempty(R2_units_bestModel_combined{i}{k}))
            R2_units_bestModel{i}(k)=nanmean(R2_units_bestModel_combined{i}{k});
        end
    end

    idx1=unit_preference_index{1,i};
    idx2=unit_preference_index{2,i};
    idx3=unit_preference_index{3,i};
    idx5=unit_preference_index_unilateral{1,i};

    total_signle_pref=[total_signle_pref;[single_race(idx1),single_compete(idx1),race_compete(idx1)]];
    total_race_pref=[total_race_pref;[single_race(idx2),single_compete(idx2),race_compete(idx2)]];
    total_compete_pref=[total_compete_pref;[single_race(idx3),single_compete(idx3),race_compete(idx3)]];

    aidx=find(strcmp(regions{i},myHierarchy));
    figure(60+m);hold on;
    R2_combined=[totalR{i}(idx1,1);totalR{i}(idx2,2);totalR{i}(idx3,2);totalR{i}(idx5,1)];
    scatter(aidx*ones(1,length(R2_combined)),R2_combined,25,'markerFaceColor',colors(i,:),'markerEdgeColor','none');
    boxplot2(R2_combined,aidx,'barwidth',0.5,'color',[0,0,0]);alpha 0.5

    %% plot unit percentage and sign test results
    sngl=totalAIC{i}([idx1;idx2;idx3],1);
    dual=nanmin(totalAIC{i}([idx1;idx2;idx3],[2,3]),[],2);
    acc_type=[ones(sum(bestModel==1),1);-1*ones(sum(bestModel==2 | bestModel==3),1)];
    [p(i),~] = signtest(acc_type);
    p(i)=p(i)*6;
    figure(1);hold on;
    xx=sum(acc_type==1)/length(bestModel);
    yy=sum(acc_type==-1)/length(bestModel);
    bar(aidx-0.2,xx,'barWidth',0.3,'faceColor',colors(i,:),'edgeColor','None')
    bar(aidx+0.2,yy,'barWidth',0.3,'faceColor','None','edgeColor',colors(i,:))
    pos_y=(max([xx,yy])+0.03);
    if(p(i)<0.05)
        line([aidx-0.2,aidx+0.2],pos_y*ones(1,2),'color',[0,0,0])
        if(p(i)<0.001)
            text(aidx,pos_y+0.03,'***');
        elseif(p(i)<0.01)
            text(aidx,pos_y+0.03,'**');
        elseif(p(i)<0.05)
            text(aidx,pos_y+0.03,'*');
        end
    end

end
h=figure(1);
axis([0,7,0,0.8])
set(h,'Position',[100,100,400,300]);
saveas(h,[imgPath,'single_dual_count_test_bilateral.pdf']);

h=figure;hold on;
plot3(total_signle_pref(:,1),total_signle_pref(:,2),total_signle_pref(:,3),'o','markerFaceColor',sel_color_single,'markerEdgeColor','none')
plot3(total_race_pref(:,1),total_race_pref(:,2),total_race_pref(:,3),'o','markerFaceColor',sel_color_race,'markerEdgeColor','none')
plot3(total_compete_pref(:,1),total_compete_pref(:,2),total_compete_pref(:,3),'o','markerFaceColor',sel_color_compete,'markerEdgeColor','none')
xlabel('single-race');ylabel('single-compete');zlabel('race-compete');
axis([-aic_lim,aic_lim,-aic_lim,aic_lim,-aic_lim,aic_lim])
view(20,10);
grid on;axis square
set(h,'Position',[100,100,300,255]);
plot2svg([imgPath,'AIC_diff_total.svg'],h)
saveas(h,[imgPath,'AIC_diff_total.pdf']);
close(h)

h=figure(10+m);
axis([0,length(myHierarchy)+1,-0.1,1])
xticks(1:length(myHierarchy));xticklabels(myHierarchy);xtickangle(45);
ylabel('R2')
set(h,'Position',[100,100,400,300])
saveas(h,[imgPath,'totalR2.pdf']);
close(h)
save([rootpath,'accSelectivity_3models'],'unit_preference_index','unit_preference_index_unilateral','R2_units_bestModel')

x_lim=[0,.3];
y_lim=[0,.4];
%% dist of connection strength across regions
single_connection=cell(1,length(myHierarchy));
dual_connection=cell(1,length(myHierarchy));
corrType='Pearson';
model_tau_single_dual=cell(2,length(myHierarchy));
corr_tau_s=zeros(2,length(myHierarchy));
combined_model_tau=cell(1,length(myHierarchy));
combined_model_autocorr=cell(1,length(myHierarchy));
for j=1 : 2
    connection=cell(length(regions),4);
    combinedTau=cell(1,4);
    combinedConn=cell(1,4);
    for i=1 : length(myHierarchy)
        aidx=find(strcmp(regions{i},myHierarchy));

        if(j==1)
            strength=A_matrix{j,i}(:,1);
            model_tau_single_dual{1,i}=[best_modelTau{i}(unit_preference_index{j,i});best_modelTau{i}(unit_preference_index_unilateral{j,i})];
            combined_model_tau{i}=[combined_model_tau{i};model_tau_single_dual{1,i}];
            combined_model_autocorr{i}=[combined_model_autocorr{i};best_modelAutocorr{i}(unit_preference_index{j,i},:);best_modelAutocorr{i}(unit_preference_index_unilateral{j,i},:)];
            S=[strength(unit_preference_index{j,i});strength(unit_preference_index_unilateral{j,i})];
            single_connection{i}=S;

            % plot tau task and intrinsic correlation ------------
            if(isempty(unit_preference_index_unilateral{j,i}))
                continue;
            end

            h=figure(2);
            fig_idx2=aidx;
            subplot(1,length(myHierarchy),fig_idx2);
            x=S;
            y=model_tau_single_dual{1,i};
            [C,P]=corr(x,y,'Rows','complete','Type',corrType);
            P=P*6;% multiple comparison correction
            corr_tau_s(1,i)=C;
            mdl=fitlm(x,y);
            plot(mdl,'marker','x','markerEdgeColor',colors(i,:),'markerSize',5);
            xlim([0.9,1.01]);ylim([0,0.31])
            xlabel('strength');ylabel('\tau-intrinsic/A')
            title(['C',num2str(C,2),' P:',num2str(P,2)]);
            axis square
            set(h,'Position',[300,0,1200,600]);

        else
            strength1=nanmean([A_matrix{2,i}(:,1,1),A_matrix{2,i}(:,2,2)],2);
            strength2=nanmean([A_matrix{3,i}(:,1,1),A_matrix{3,i}(:,2,2)],2);
            model_tau_single_dual{2,i}=[best_modelTau{i}(unit_preference_index{2,i});best_modelTau{i}(unit_preference_index{3,i})];
            combined_model_tau{i}=[combined_model_tau{i};model_tau_single_dual{2,i}];
            combined_model_autocorr{i}=[combined_model_autocorr{i};best_modelAutocorr{i}(unit_preference_index{2,i},:);best_modelAutocorr{i}(unit_preference_index{3,i},:)];
            S=[strength1(unit_preference_index{2,i});strength2(unit_preference_index{3,i})];
            dual_connection{i}=S;

            % plot tau task and intrinsic correlation ------------
            if(isempty(unit_preference_index{2,i}) && isempty(unit_preference_index{3,i}))
                continue;
            end

            h=figure(3);
           
            fig_idx2=aidx;
            subplot(1,length(myHierarchy),fig_idx2);
            x=S;
            y=model_tau_single_dual{2,i};

            [C,P]=corr(x,y,'Rows','complete','Type',corrType);
            P=P*3;% multiple comparison correction
            corr_tau_s(2,i)=C;
            mdl=fitlm(x,y);
            plot(mdl,'marker','x','markerEdgeColor',colors(i,:),'markerSize',5);
            xlim([0.95,1.01]);ylim([0,0.25])
            xlabel('strength');ylabel('\tau-intrinsic/A')
            title(['C',num2str(C,2),' P:',num2str(P,2)]);
            axis square
            set(h,'Position',[300,0,1200,600]);
        end
    end
end

%% fitting tau
mean_model_tau=[];
std_model_tau=[];
lags=bins(2:end-1);
shuffel=100;
murrayTau=zeros(length(myHierarchy),shuffel);

for i=1 : length(myHierarchy)
    valid=find(~isnan(combined_model_autocorr{i}(:,1)));
    totalCorr=combined_model_autocorr{i}(valid,:);
    Lag=repmat(lags,size(totalCorr,1),1);
    ucount=size(totalCorr,1);
    for j=1 : shuffel
        selIdx=1+round((ucount-1)*rand(1,ucount));
        selCorr=totalCorr(selIdx,:);
        fitR = fit_tau(selCorr(:),Lag(:),1,lag);
        murrayTau(i,j)=fitR.tau;
    end

    mean_model_tau(i)=nanmean(murrayTau(i,:));
    std_model_tau(i)=1.98*nanstd(murrayTau(i,:))/sqrt(shuffel);
end

save([rootpath,'modelTauMurray'],'murrayTau')

[V,I]=sort(mean_model_tau);
sorted_regions=regions(I);

for i=1 : length(sorted_regions)
    aidx=find(strcmp(sorted_regions{i},regions));
    figure(4);hold on;boxplot2(murrayTau(aidx,:)',i,'barwidth',0.5,'color',colors(aidx,:));alpha 0.5
    figure(5);hold on;
    scatter(i*ones(1,length(best_modelTau{aidx})),best_modelTau{aidx},25,'markerFaceColor',colors(aidx,:),'markerEdgeColor','none');
end

h=figure(4);
N=6;
for i=1 : length(sorted_regions)-1
    p = ranksum(murrayTau(I(i+1),:), murrayTau(I(i),:),'Tail','right');
    vals=[murrayTau(I(i+1),:),murrayTau(I(i),:)];
    if(N*p<0.05)
        yy=max(mean_model_tau([I(i),I(i+1)]))+0.02;
        line([i,i+1],[yy,yy],'color','k');
        if(N*p<0.001)
            text(i+0.5,yy+0.005,'***')
        elseif(N*p<0.01)
            text(i+0.5,yy+0.005,'**')
        elseif(N*p<0.05)
            text(i+0.5,yy+0.005,'*')
        end
    end
end

xticks(1:length(sorted_regions));
xticklabels(sorted_regions);
xtickangle(45)
axis square
xlim([0,length(sorted_regions)+1])
ylim([0.04,0.18])
saveas(h,[imgPath,'tau_hierarchy.pdf']);

h=figure(5);
xticks(1:length(sorted_regions));
xticklabels(sorted_regions);
xtickangle(45)
axis square
xlim([0,length(sorted_regions)+1])
ylim([0,0.33])
alpha 0.5
saveas(h,[imgPath,'tau_distribution.pdf']);
h=figure(2);
set(h,'PaperOrientation','landscape')
print(h,[imgPath,'tau_connection_',method{1},'_unilateral'],'-dpdf','-bestfit')

h=figure(3);
set(h,'PaperOrientation','landscape')
print(h,[imgPath,'connection_strength_bilateral_combined_race_compete',],'-dpdf','-bestfit')

colors2=[42, 157, 143;233, 196, 106;216, 17, 89]/255;
totalCount=zeros(length(myHierarchy),4);
for i=1 : length(myHierarchy)
    aidx=find(strcmp(regions{i},myHierarchy));
    totalCount(aidx,:)=[unit_per_model_unilateral(i,1),unit_per_model(i,:)];
end

h=figure;
bar(totalCount,'stacked', 'BarWidth', 0.7);
xticks(1:length(myHierarchy));xticklabels(myHierarchy);
xtickangle(45);
xlim([0,length(myHierarchy)+1]);ylim([0,1.1])
axis square
set(h,'Position',[100,100,300,300])
saveas(h,[imgPath,'total_accumulator_percentage.pdf']);

