import ssm
import ssm.extensions.mp_srslds.emissions_ext as mp_emission
import ssm.extensions.mp_srslds.transitions_ext_2 as mp_transition
import ssm.init_state_distns as isd
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import autograd.numpy.random as npr
import seaborn as sns
from ssm.plots import gradient_cmap, white_to_color_cmap
import pickle
import os
from os import listdir
from os.path import isfile, join
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D
from ssm.util import softplus
from ssm.primitives import blocks_to_full, convert_lds_to_block_tridiag
from ssmdm.accumulation import Accumulation, LatentAccumulation
import copy
import multiprocessing

npr.seed(12345)

filepath='../dataset/Mice/result/Accumulator/spikes_subsample_postWheel/'

regions=['frontal']

sessions=np.linspace(1,39,39)
# sessions=[18]
ignore_latency=False
learnV=0
learnA=1
learnA_state2_3=0
constantEmission=False

bound_offsets=[0.3,0.4]
taus=[50,100]
init_A1_vals=[0.95]
init_A2_vals=[0]
init_V_vals=[0.02,0.03,0.04]
init_sigma_vals=[0.001,0.002]
scale=1

init_params=[]
file_postfix=[]
for i in range(0,len(init_A1_vals)):
    for j in range(0,len(init_A2_vals)):
        for k in range(0,len(init_V_vals)):
            for s in range(0,len(init_sigma_vals)):
                for b in range(len(bound_offsets)):
                    for t in range(len(taus)):
                        param={'a1':init_A1_vals[i], 'a2':init_A2_vals[j], 'v':init_V_vals[k], 'sigma':init_sigma_vals[s],'bound_offset':bound_offsets[b],'tau':taus[t]}
                        init_params.append(param)
                        postfix="_A1_"+str(init_A1_vals[i]).replace(".","_")
                        postfix=postfix+"_A2_"+str(abs(init_A2_vals[j])).replace(".","_")
                        postfix=postfix+"_V_"+str(abs(init_V_vals[k])).replace(".","_")
                        postfix=postfix+"_sigma_"+str(abs(init_sigma_vals[s])).replace(".","_")
                        postfix=postfix+"_bound_"+str(abs(bound_offsets[b])).replace(".","_")
                        postfix=postfix+"_tau_"+str(abs(taus[t])).replace(".","_")
                        postfix=postfix+"_scale_"+str(scale)
                        file_postfix.append(postfix)

# parameter settings -----------------------------------------------------------
dt=0.001

K=3
# D=2
M=2 #dimension of the inputs
stateNo=1 #dimesion of each hidden state

l2_penalty_V=0
l2_penalty_b=0
l2_penalty_x1u1=0
l2_penalty_x1u2=0
l2_penalty_x2u1=0
l2_penalty_x2u2=0

l2_penalty_A=0
l2_penalty_x1x1=0
l2_penalty_x1x2=0
l2_penalty_x2x1=0
l2_penalty_x2x2=0


for region in regions:
    infilepath=filepath+region+'/'
    if(ignore_latency):
        outpath=infilepath+'_results_single_collaps_ignoreLat/'
    else:
        outpath=infilepath+'_results_single_collaps/'

    if(os.path.exists(outpath) == 0):
        os.mkdir(outpath)
        
    onlyfiles = [f for f in listdir(infilepath) if isfile(join(infilepath, f))]
    for n in range(len(onlyfiles)):
        if('.mat' not in onlyfiles[n]):
            continue
        
        name=onlyfiles[n]
        infilepath2=infilepath+name
        nameparts=name.split(".")
        outpath2=outpath+nameparts[0]+'/'
        if(os.path.exists(outpath2) == 0):
            os.mkdir(outpath2)

        fitFiles = [f for f in listdir(outpath2) if isfile(join(outpath2, f))]
        dataset=scipy.io.loadmat(infilepath2)
        spikes=dataset['spikes']
        right_evidence=dataset['right']
        left_evidence=dataset['left']
        postWheelLen=dataset['postWheelLen']
        times=dataset['times'][0]
        dt=times[0][0,1]-times[0][0,0]
        right_evidence=np.array([re[0] for re in right_evidence])
        left_evidence=np.array([le[0] for le in left_evidence])
        postWheelSmpl=int(postWheelLen/dt)

        evidence=right_evidence-left_evidence
        latency_smpl=dataset['latency_smpl']
        latency_smpl=latency_smpl[0,0]
        sess=dataset['sess']
        ss,trialNo=spikes.shape
        if(sess not in sessions):
            continue

        # data preparation -------------------------------------------------------------
        N_vec=[]
        D_vec=[]
        D_vec.append(stateNo)

        #read number of neurons for each regions
        data=spikes[0,0]
        N,T=data.shape
        N_vec.append(N)

        initY=[]
        initY_evidence=[]
        initD=[]
        U=[]
        Y=[]
        for t in range(trialNo):
            y=[]
            init_y=[]
            init_d=[]

            un,smpl = np.shape(spikes[0,t])
            data=spikes[0,t].transpose()

            data=data.astype(np.int)
            smpl,N=np.shape(data)

            if(postWheelSmpl>0):
                init_y=np.mean(data[-1*postWheelSmpl-10:-1*postWheelSmpl,:],axis=0)
            else:
                init_y=np.mean(data[-10:,:],axis=0)

            init_d=np.mean(data[:latency_smpl,:],axis=0)
            
            if(ignore_latency):
                Y.append(data)
                ev=np.ones((smpl,1))*evidence[t]
                times=dt*np.linspace(1,smpl,smpl)
                u=np.hstack((ev,times[:,None]))
                U.append(u)
            else:
                Y.append(data[latency_smpl:,:])
                ev=np.ones((smpl-latency_smpl,1))*evidence[t]
                times=dt*np.linspace(1,smpl-latency_smpl,smpl-latency_smpl)
                u=np.hstack((ev,times[:,None]))
                U.append(u)


            initY_evidence.append(init_y)
 
            initD.append(init_d)

            D=np.sum(D_vec)
            D_vec_cumsum = np.concatenate(([0], np.cumsum(D_vec)))


        # initialize model -------------------------------------------------------------
        poiss_comp_emissions=mp_emission.PoissonCompoundEmissions(
            N=np.sum(N_vec),K=K ,D=np.sum(D_vec), M=M ,D_vec=D_vec, N_vec=N_vec,link='softplus',constantEmission=constantEmission)

        for param in range(len(init_params)):
            outfile=nameparts[0]+file_postfix[param]+'.pkl'
            if(outfile in fitFiles):
                continue
            
            sro_trans=mp_transition.singleAccumulatorMultipop_nonlinearCollapsingBound(K=K,D=np.sum(D_vec),M=M, l2_penalty_similarity=0, l1_penalty=0, scale=scale, bound_offset=init_params[param]['bound_offset'], tau=init_params[param]['tau']) 

            #Create initial ground truth model, that we will modify
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
            init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(K-1))*np.ones(K-1))))

            rslds = ssm.SLDS(N=np.sum(N_vec), K=K, M=M, D=np.sum(D_vec),
                            dynamics="gaussian_independant",
                            emissions=poiss_comp_emissions,
                            transitions=sro_trans,
                            dynamics_kwargs=dict(l2_penalty_A=l2_penalty_A, l2_penalty_V=l2_penalty_V, learn_V=learnV, learn_A=learnA, learnA_state2_3=learnA_state2_3))

            # initialization
            d_init = (np.mean(initD,axis=(0)) / dt).reshape((1,np.sum(N_vec)))

            C_init_e = np.mean(initY_evidence, axis=(0)).reshape((1,np.sum(N_vec))) / dt - d_init
            C_init=np.zeros((1,np.sum(N_vec),D))
            C_init[0,:,0]=C_init_e
            rslds.emissions.Cs = C_init
            rslds.emissions.ds = d_init

            # model fitting ----------------------------------------------------------------
        
            Vs=np.zeros((K,1,2))
            Vs[0,0,0]=init_params[param]['v']
            rslds.dynamics.Vs=Vs
            rslds.dynamics.bs=np.zeros((K,D))
            Sigma=init_params[param]['sigma']*np.tile(np.ones((1,1)),(K,1,1))
            rslds.dynamics.Sigmas=Sigma
            rslds.dynamics.Sigmas_init=rslds.dynamics.Sigmas
            A=np.tile(np.ones((1,1)),(K,1,1))
            A[0]=init_params[param]['a1']
            rslds.dynamics.As=A
            try:
                q_elbos_ar, q_ar = rslds.fit(Y, inputs=U, method="laplace_em",
                                                                            variational_posterior="structured_meanfield", 
                                                                            continuous_optimizer='newton',
                                                                            initialize=False, 
                                                                            num_init_restarts=10,
                                                                            num_iters=50, 
                                                                            alpha=0.3, variational_posterior_kwargs={"initial_variance":1e-5})

                print("End of fitting "+str(param))
                init_A1=init_params[param]['a1']
                init_A2=init_params[param]['a2']
                init_V=init_params[param]['v']
                init_sigma=init_params[param]['sigma']

                lat=latency_smpl
                if(ignore_latency):
                    lat=0

                model={'dt':dt,'init_A1':init_A1,'init_A2':init_A2,'init_V':init_V,'init_sigma':init_sigma,'Y':Y , 'rslds':rslds, 'q_ar':q_ar, 'q_elbos_ar':q_elbos_ar,'inputs':U, 'K':K, "D_vec":D_vec, 'N_vec':N_vec, 'latency':lat}
                partName=name.split(".mat")
                outfile=outpath2+nameparts[0]+file_postfix[param]+'.pkl'
                with open(outfile, 'wb') as fhandle:
                    pickle.dump(model, fhandle)

            except:
                print("Error in fitting process")
