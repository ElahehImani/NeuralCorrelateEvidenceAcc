clc;clear;close all;
root='..\';
addpath(genpath([root,'NeuralCorrelateEvidenceAcc']))
addpath(genpath([rootpath,'npy-matlab-master\']))
addpath(genpath([rootpath,'steinmetz-et-al-2019-master\']))
%%
datapath='..\dataset\Mice\';
rootpath=[datapath,'9598406\spikeAndBehavioralData\'];
outpath=[datapath,'result\'];

regCluster=1;% use clusters of regions
clusterName='clusterArea_paper';
clusterRegPath=[outpath,clusterName];
load(clusterRegPath);
stim_conditions={'Right','Left','NoGo','EqualContrast'};
res_conditions=[-1,1,0]; %-1: right, 1: left, 0: nogo
dir_info=dir(rootpath);
task_title='ContrastDetection';
rule='contrast';
post_stim=1;
pre_stim=.2;

annotate_value=1;
annotate_op='>=';
file_postfix='total'; %total, MUA, good

if(regCluster)
    file_postfix=[file_postfix,'_',clusterName];
end

outpath=[outpath,file_postfix,'\'];
if(~exist(outpath,'dir'))
    mkdir(outpath);
end
Trial_len=[];
sess=1;
    
for i=1 : length(dir_info)
    if(dir_info(i).isdir==0 || strcmp(dir_info(i).name,'.') || strcmp(dir_info(i).name,'..'))
        continue;
    end
    sesPath=[rootpath,dir_info(i).name];
    sessData = loadSession(sesPath);
    outpath2=[outpath,dir_info(i).name,'\'];
    if(~exist(outpath2,'dir'))
        mkdir(outpath2);
    end
    
    %% trials info --------------------------------------------------
    trial_len=length(sessData.trials.response_times);
    if(~isempty(Trial_len))
        trial_len=Trial_len;
    end
    trials=1:trial_len;
    rules=[];
    tasks=[];
    for c=1 : trial_len
        rules{c}=rule;
        tasks{c}=task_title;
    end
    
    feedback=sessData.trials.feedbackType;
    outcome=[];
    for c=1 : trial_len
        if(feedback(c)==1)
            outcome{c}='correct';
        else
            outcome{c}='choiceErr';
        end
    end
    
    correct=feedback==1;
    correct=correct(1:trial_len);
    
    trialRefTimes=sessData.trials.intervals(:,1);
    trialRefTimes=trialRefTimes(1:trial_len);
    
    trialEndTimes=sessData.trials.intervals(:,2);
    trialEndTimes=trialEndTimes(1:trial_len);
    ruleCue=[];
    expectedResponse=[];
    preRuleCue=[];
    preExpectedResponse=[];
    preChosenResponse=[];
    wheelOn=[];
    for c=1 : trial_len
        if(c>1)
            preRuleCue{c}=ruleCue{c-1};
            preExpectedResponse(c)=expectedResponse(c-1);
        else
            preRuleCue{1}='';
            preExpectedResponse(1)=NaN;
        end
        if(sessData.trials.visualStim_contrastRight(c)>sessData.trials.visualStim_contrastLeft(c))
            ruleCue{c}='Right';
            expectedResponse(c)=-1;
        elseif(sessData.trials.visualStim_contrastRight(c)<sessData.trials.visualStim_contrastLeft(c))
            ruleCue{c}='Left';
            expectedResponse(c)=1;
        elseif(sessData.trials.visualStim_contrastRight(c)==0 && sessData.trials.visualStim_contrastLeft(c)==0)
            ruleCue{c}='NoGo';
            expectedResponse(c)=0;
        else 
            ruleCue{c}='EqualContrast';
            if(feedback(c))
                expectedResponse(c)=sessData.trials.response_choice(c);
            else
                if(sessData.trials.response_choice(c)~=0)
                    expectedResponse(c)=-1*sessData.trials.response_choice(c);
                else
                    if(rand>=.5)
                        expectedResponse(c)=-1;
                    else
                        expectedResponse(c)=1;
                    end
                end
            end
        end
        
        %% wheel onset
        widx=find(sessData.wheelMoves.intervals(:,1)>=sessData.trials.visualStim_times(c) & ...
        sessData.wheelMoves.intervals(:,1)<=sessData.trials.response_times(c) & ...
        (sessData.wheelMoves.type==1 | sessData.wheelMoves.type==2));

        if(~isempty(widx))
            wheelOn(c)=sessData.wheelMoves.intervals(widx(1),1);
        else
            wheelOn(c)=NaN;
        end
    end
    
    stim_cond_idx=zeros(1,trial_len);
    for c=1 : length(stim_conditions)
        stim_cond_idx(strcmp(ruleCue,stim_conditions{c}))=c;
    end
    
    chosenResponse=sessData.trials.response_choice;
    chosenResponse=chosenResponse(1:trial_len);
    preChosenResponse=[NaN;chosenResponse(1:end-1)];
    
    res_cond_idx=zeros(1,trial_len);
    for c=1 : length(res_conditions)
        res_cond_idx(chosenResponse==res_conditions(c))=c;
    end
    
    cueOn=sessData.trials.goCue_times;
    cueOn=cueOn(1:trial_len);
    
    stimOn=sessData.trials.visualStim_times;
    stimOn=stimOn(1:trial_len);
    
    responseTime=sessData.trials.response_times;
    responseTime=responseTime(1:trial_len);
    
    feedbackTime=sessData.trials.feedback_times;
    feedbackTime=feedbackTime(1:trial_len);
    
    leftContrastLevel=sessData.trials.visualStim_contrastLeft(1:trial_len);
    rightContrastLevel=sessData.trials.visualStim_contrastRight(1:trial_len);
    
    trialInfo = table(tasks',trials',outcome',correct,trialRefTimes,trialEndTimes,rules',ruleCue',preRuleCue',...
        expectedResponse',preExpectedResponse',chosenResponse,preChosenResponse,cueOn,stimOn,wheelOn',responseTime,feedbackTime,rightContrastLevel,leftContrastLevel,...
        'VariableNames',{'task','trial','outcome','correct','trialRefTimes','trialEndTimes','rule', ...
        'ruleCue','preRuleCue','expectedResponse','preExpectedResponse','chosenResponse','preChosenResponse','cueOn','stimOn','wheelOn','responseTime','feedbackTime',...
        'rightContrastLevel','leftContrastLevel'});
    

    %% spike info
    spike_clusters=sessData.spikes.clusters;
    cluster_probes=sessData.clusters.probes;
    annotation_cluster=sessData.clusters.x_phy_annotation;
    switch(annotate_op)
        case '>'
            valid_idx=find(annotation_cluster>annotate_value);
        case '>='
            valid_idx=find(annotation_cluster>=annotate_value);
        case '='
            valid_idx=find(annotation_cluster==annotate_value);
    end
    
    annotation=annotation_cluster(valid_idx);
    
    valid_cluster_probe=cluster_probes(valid_idx);
    cluster_valid_idx=valid_idx-1;
    spike_times=sessData.spikes.times;
    clusterSpikeTimes=[];
    for c=1 : length(cluster_valid_idx)
        spike_cluster_idx=find(spike_clusters==cluster_valid_idx(c));
        clusterSpikeTimes{c}=spike_times(spike_cluster_idx);  
    end
        
     %% unitInfo ---------------------------
    probes=sessData.channels.probe;
    unit=sessData.channels.site;
    trialIdxRange=[];
    trialRange=[];
    cluster_unit=[];
    cluster_area_unit=[];
    cluster_subArea_unit=[];
    for c=1 : length(cluster_valid_idx)
        trialIdxRange{c}=[1,trial_len];
        trialRange{c}=[1,trial_len];
        cluster_unit(c)=cluster_valid_idx(c);
        cluster_area_unit{c}=sessData.channels.brainLocation.allen_ontology(sessData.clusters.peakChannel(valid_idx(c)),:);
        cluster_area_unit{c}=replace(cluster_area_unit{c},' ','');
        cluster_subArea_unit{c}=cluster_area_unit{c};
        if(regCluster)
            cluster_area_unit{c}=getClusterName(clusterArea,cluster_area_unit{c});
        end
    end

    session=sess*ones(1,length(cluster_valid_idx));
    
    unitInfo = table(session',valid_cluster_probe,cluster_unit',trialIdxRange',trialRange',cluster_area_unit',cluster_subArea_unit',annotation,...
        'VariableNames',{'session','electrode','unit','trialIdxRange','trialRange','area','subArea','annotation','cellDepth'});

    save([outpath2,'BasicInfo'],'trialInfo','unitInfo','clusterSpikeTimes',...
        'annotate_value','annotate_op','-v7.3');    
    
    disp(['sess ',num2str(sess),'/',num2str(length(dir_info))])
    sess=sess+1;
end
