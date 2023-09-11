from cmath import nan
import pickle
from matplotlib import pyplot as plt
from numpy.lib import math
from scipy.sparse.construct import rand
from ssm.plots import gradient_cmap, white_to_color_cmap
import seaborn as sns
import numpy as np
import os
import os
from scipy.io import savemat,loadmat
import os
from os import listdir
from os.path import isfile, join
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import ssm.init_state_distns as isd

regions=['frontal']
method='single_collaps' #single,compete,race,single_collaps
shuffel=100
cont=np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
filepath='../dataset/Accumulator/spikes_subsample_postWheel/'
filepath_out='../dataset/Accumulator/spikes_subsample_postWheel_bestFit/'

ignore_latency=False
saveFig=0
flen=0.05
R2_length=0.25
cmp=plt.cm.get_cmap('bwr',9)
colors=cmp(range(9))

# setting -----------------------------------------------------------------------
sns.set_style("white")
sns.set_context("talk")
sns.set_style('ticks',{"xtick.major.size":8,"ytick.major.size":8})


for r in range(len(regions)):#loop over regions
    region=regions[r]
    regPath=filepath+region

    outpath2=regPath+'/_results_'+method+'/'
    
    resPath=filepath_out+region+'/_results_'+method+'/'+'_BestLogLike/'
    if(os.path.exists(resPath)==0):
        os.mkdir(resPath)

    onlyfiles_raw = [f for f in listdir(regPath) if isfile(join(regPath, f))]

    for n in range(len(onlyfiles_raw)):
        nameparts_raw=onlyfiles_raw[n].split(".mat")
        EV_Path=regPath+'/_results_'+method+'/'+nameparts_raw[0]+'/_single_EV/'
        if(os.path.exists(EV_Path)==0):
            continue

        onlyfiles = [f for f in listdir(EV_Path) if isfile(join(EV_Path, f))]
        total_LL=[]
        name_files=[]
        rawData=scipy.io.loadmat(regPath+'/'+nameparts_raw[0]+'.mat')
        RT=rawData['RT']
        fitNames=[]
        for n2 in range(len(onlyfiles)):#loop over populations
            nameparts=onlyfiles[n2].split("_EV.mat")
            
            # load log likelihood --------------------------------------
            LL_data=scipy.io.loadmat(EV_Path+onlyfiles[n2])
            fitNames.append(nameparts[0])
            total_LL.append(LL_data['logLike'][0])

        # find best LL -----------------------------------------------
        max_idx=total_LL.index(np.nanmax(total_LL))
        logLike=total_LL[max_idx][0]
        outfile_mat_single=resPath+nameparts_raw[0]+'_best_EV.mat'

        model = pickle.load(open(regPath+'/_results_'+method+'/'+nameparts_raw[0]+'/'+fitNames[max_idx]+'.pkl', "rb" ))

        rslds=model['rslds']
        U=model['inputs']
        q_ar=model['q_ar']
        spikes=model['Y']
        (smpl,unitNo)=np.shape(spikes[0])
        latency_smpl=model['latency']
        dt=model['dt']
        Y_main=model['Y']
        medRt_smpl_totalCond=int(np.median(RT)/dt-latency_smpl)
        flen2=int(flen/dt)
        filt=1/flen2*np.ones((flen2,))

        totalR2_pop=np.empty((shuffel,))
        totalR2_pop.fill(np.nan)
        totalR2_units=np.empty((unitNo,shuffel))
        totalR2_units.fill(np.nan)

        #we set the init dist to sample correctly ----------------------------
        init_state_distn = isd.InitialStateDistribution(rslds.K, rslds.D, M=rslds.M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(rslds.K-1))*np.ones(rslds.K-1))))
        rslds.init_state_distn=init_state_distn
        
        init_A1_val=model['init_A1']
        init_A2_val=model['init_A2']
        init_V_val=model['init_V']
        init_Sigma_val=model['init_sigma']

        trialNo=len(U)
        evidence=np.zeros((trialNo,))

        # bootsrapping trials within each contrast level ----------------------
        for t in range(trialNo):
            if(method=='single'):
                evidence[t]=U[t][0]
            elif(method=='single_collaps'):
                evidence[t]=U[t][0,0]
            else:
                evidence[t]=U[t][0,0]-U[t][0,1]

        bootstrapped_idx=[]
        for e in range(len(cont)):
            idx=np.where(evidence==cont[e])[0]
            if(len(idx)==0):
                continue
            
            bt_idx=np.round(np.random.rand(len(idx))*(len(idx)-1))
            bt_idx=bt_idx.astype(int)
            bootstrapped_idx=np.hstack((bootstrapped_idx,idx[bt_idx]))

        bootstrapped_idx=bootstrapped_idx.astype(int)

        # compute R2 --------------------------------------------
        Y_bootstrapped=[]
        RT_bootstrapped=np.zeros(np.shape(RT))
        evidence_bootsrapped=np.zeros((trialNo,1))
        U_bootsrapped=[]
        log_likes_shuffels=np.empty((shuffel,trialNo))
        maxR2=-10
        for s in range(shuffel):
            try:

                Y=[]
                maxLen=0
                mask=[]
                log_likes=0
                for t in range(trialNo):                
                    idx=bootstrapped_idx[t]
                    if(method=='single'):
                        evidence_bootsrapped[t]=U[idx][0]
                    elif(method=='single_collaps'):
                        evidence_bootsrapped[t]=U[idx][0,0]
                    else:
                        evidence_bootsrapped[t]=U[idx][0,0]-U[idx][0,1]
                    
                    u=U[idx]
                    U_bootsrapped.append(u)
                    Y_bootstrapped.append(Y_main[idx])
                    RT_bootstrapped[t]=RT[idx]
                    mask.append(np.ones_like(Y_main[idx], dtype=bool))

                    (smpl,dim)=u.shape
                    z, x, y = rslds.sample(smpl, input=u)
                    Y.append(y)
                    maxLen=max(maxLen,smpl)                    

                #average within each evidence level                    
                maxLen=int(maxLen)
                fr=np.empty((len(cont),unitNo,maxLen))
                fr.fill(np.nan)
                fr_model=np.empty((len(cont),unitNo,maxLen))
                fr_model.fill(np.nan)

                max_samples=0
                medRt_smpl=np.zeros((len(cont),))
                for e in range(len(cont)):
                    idx=np.where(evidence_bootsrapped==cont[e])[0]
                    if(len(idx)==0):
                        continue
                
                    medRt_smpl[e]=int(np.median(RT_bootstrapped[idx])/dt-latency_smpl)
                    fr_model_tmp=np.empty((len(idx),unitNo,maxLen))
                    fr_tmp=np.empty((len(idx),unitNo,maxLen))
                    mask=np.ones((maxLen,))

                    for i in range(len(idx)):
                        smpl,unitNo=np.shape(Y[idx[i]])
                        for u in range(unitNo):
                            #filtering spikes
                            fr_model_tmp[i,u,:smpl]=signal.lfilter(filt,1,Y[idx[i]][:,u])
                            fr_tmp[i,u,:smpl]=signal.lfilter(filt,1,Y_bootstrapped[idx[i]][:,u])
                            mask[smpl:]=0

                    valid_idx=np.nonzero(mask)[0]
                    max_valid_idx=int(valid_idx[-1])
                    max_samples=max(max_samples,max_valid_idx)

                    #we only consider samples lower than median RT
                    fr_model[e,:,:medRt_smpl_totalCond]=np.nanmean(fr_model_tmp[:,:,:medRt_smpl_totalCond],axis=0)
                    fr[e,:,:medRt_smpl_totalCond]=np.nanmean(fr_tmp[:,:,:medRt_smpl_totalCond],axis=0)

                newLen=np.shape(fr)[2]

                #compute R2 neuron-level
                totalMean=np.nanmean(fr,axis=(0,2))
                totalMean=totalMean[:,None]
                totalMean=np.tile(totalMean,(len(cont),1,newLen))
                totalVar=np.nansum(np.power(fr-totalMean,2),axis=(0,2))
                res=np.nansum(np.power(fr-fr_model,2),axis=(0,2))
                unitR2=1-np.divide(res,totalVar)

                #compute R2 pop-level
                avgFr=np.nanmean(fr,axis=1)
                avgFr_model=np.nanmean(fr_model,axis=1)
                totalMean_avg=np.nanmean(avgFr,axis=(0,1))
                totalVar_avg=np.nansum(np.power(avgFr-totalMean_avg,2),axis=(0,1))

                res=np.nansum(np.power(avgFr-avgFr_model,2),axis=(0,1))
                popR2=1-res/totalVar_avg
                
                totalR2_pop[s]=popR2
                totalR2_units[:,s]=unitR2
                trained_A=rslds.dynamics.As
                trained_V=rslds.dynamics.Vs
                trained_sigma=rslds.dynamics.Sigmas
                
                print("finish sampling "+nameparts[0]+" sh"+str(s))

                times=dt*np.linspace(1,newLen,newLen)
                times=times+dt*latency_smpl

            except:
                print("Error in sampling")

        
        ## save single files
        mdic={"totalR2_pop":totalR2_pop, 'totalR2_units':totalR2_units,'trained_A':trained_A,'trained_V':trained_V, 'trained_Sigma':trained_sigma,"logLike":logLike}
        savemat(outfile_mat_single,mdic)
        print("finish sampling "+nameparts[0])
    