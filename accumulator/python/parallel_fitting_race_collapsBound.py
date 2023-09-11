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

filepath='../dataset/Accumulator/spikes_subsample_postWheel/'

regions=['frontal']

sessions=np.linspace(1,39,39)
ignore_latency=False
learnV=0
learnA=1
learnA_state2_3=0
constantEmission=False

bound_offsets=[0.3,0.4,0.5]
taus=[50,100]
init_A1_vals=[.95]
init_A2_vals=[0]
init_V1_vals=[0.01,0.02,0.03]
init_V2_vals=[-0.01,-0.02,-0.03]
init_sigma_vals=[0.0025]

init_params=[]
file_postfix=[]
for i in range(0,len(init_A1_vals)):
    for j in range(0,len(init_A2_vals)):
        for k in range(0,len(init_V1_vals)):
            for k2 in range(0,len(init_V2_vals)):
                for k3 in range(0,len(init_V2_vals)):
                    for s in range(0,len(init_sigma_vals)):
                        for b in range(len(bound_offsets)):
                            for t in range(len(taus)):
                                param={'a1':init_A1_vals[i], 'a2':init_A2_vals[j], 'v1':init_V1_vals[k], 'v12':init_V2_vals[k2], 'v21':init_V2_vals[k3], 'sigma':init_sigma_vals[s],'bound_offset':bound_offsets[b],'tau':taus[t]}
                                init_params.append(param)
                                postfix="_A1_"+str(init_A1_vals[i]).replace(".","_")
                                postfix=postfix+"_A2_"+str(abs(init_A2_vals[j])).replace(".","_")
                                postfix=postfix+"_V1_"+str(abs(init_V1_vals[k])).replace(".","_")
                                postfix=postfix+"_V12_"+str(abs(init_V2_vals[k2])).replace(".","_")
                                postfix=postfix+"_V21_"+str(abs(init_V2_vals[k3])).replace(".","_")
                                postfix=postfix+"_sigma_"+str(abs(init_sigma_vals[s])).replace(".","_")
                                postfix=postfix+"_bound_"+str(abs(bound_offsets[b])).replace(".","_")
                                postfix=postfix+"_tau_"+str(abs(taus[t])).replace(".","_")
                                file_postfix.append(postfix)

# parameter settings -----------------------------------------------------------
dt=0.001

K=5
# D=2
M=3 #dimension of the inputs
stateNo=2 #dimesion of each hidden state
scale=1

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
    outpath=infilepath+'_results_race_collaps/'

    if(os.path.exists(outpath) == 0):
        os.mkdir(outpath)
        
    onlyfiles = [f for f in listdir(infilepath) if isfile(join(infilepath, f))]
    for n in range(len(onlyfiles)):
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
        initY_right=[]
        initY_left=[]
        initD=[]
        U=[]
        Y=[]
        for t in range(trialNo):
            y=[]
            init_y=[]
            init_d=[]

            un,smpl = np.shape(spikes[0,t])
            data=spikes[0,t].transpose()

            ev=right_evidence[t]-left_evidence[t]
            u=np.zeros((smpl,2))
            u=np.hstack((np.ones((smpl,1))*ev,np.ones((smpl,1))*(-1*ev)))
            
            data=data.astype(np.int)
            smpl,N=np.shape(data)

            if(postWheelSmpl>0):
                init_y=np.mean(data[-1*postWheelSmpl-10:-1*postWheelSmpl,:],axis=0)
            else:
                init_y=np.mean(data[-10:,:],axis=0)

            init_d=np.mean(data[:latency_smpl,:],axis=0)
            
            if(ignore_latency):
                Y.append(data)
                times=dt*np.linspace(1,smpl,smpl)
                t_tmp=np.hstack((u,times[:,None]))
                U.append(t_tmp)
            else:
                Y.append(data[latency_smpl:,:])
                times=dt*np.linspace(1,smpl,smpl)
                t_tmp=np.hstack((u,times[:,None]))
                U.append(t_tmp[latency_smpl:,:])

            if(evidence[t]>0):
                initY_right.append(init_y)
            else:
                initY_left.append(init_y)

            initD.append(init_d)

            D=np.sum(D_vec)
            D_vec_cumsum = np.concatenate(([0], np.cumsum(D_vec)))


        # initialize model -------------------------------------------------------------
        poiss_comp_emissions=mp_emission.PoissonCompoundEmissions(
            N=np.sum(N_vec),K=K ,D=np.sum(D_vec), M=M ,D_vec=D_vec, N_vec=N_vec,link='softplus',constantEmission=constantEmission)

        for param in range(len(init_params)):
        # for param in range(30,40):
            outfile=nameparts[0]+file_postfix[param]+'.pkl'
            if(outfile in fitFiles):
                continue
            sro_trans=mp_transition.dualAccumulatorMultipopCompete_collapsBound(K=K,D=np.sum(D_vec),M=M, l2_penalty_similarity=0, l1_penalty=0, scale=scale, bound_offset=init_params[param]['bound_offset'], tau=init_params[param]['tau']) 

            rslds = ssm.SLDS(N=np.sum(N_vec), K=K, M=M, D=np.sum(D_vec),
                            dynamics="gaussian_independant",
                            emissions=poiss_comp_emissions,
                            transitions=sro_trans,
                            dynamics_kwargs=dict(l2_penalty_A=l2_penalty_A, l2_penalty_V=l2_penalty_V, learn_V=learnV, learn_A=learnA, learnA_state2_3=learnA_state2_3))

            # initialization
            d_init = (np.mean(initD,axis=(0)) / dt).reshape((1,np.sum(N_vec)))

            C_init_right = np.mean(initY_right, axis=(0)).reshape((1,np.sum(N_vec))) / dt - d_init
            C_init_left = np.mean(initY_left, axis=(0)).reshape((1,np.sum(N_vec))) / dt - d_init
            C_init=np.zeros((1,np.sum(N_vec),D))
            C_init[0,:,0]=C_init_right
            C_init[0,:,1]=C_init_left
            rslds.emissions.Cs = C_init
            rslds.emissions.ds = d_init

            J0_diag = np.concatenate((l2_penalty_A * np.ones(D),
                                    l2_penalty_V * np.ones(M),
                                    l2_penalty_b * np.ones(1)))
                                    
            J0 = np.tile(np.diag(J0_diag)[None, :, :], (K, 1, 1))
            dd=int(D/2)

            J0[:,0:dd,D]=l2_penalty_x1u1
            J0[:,0:dd,D+1]=l2_penalty_x1u2
            J0[:,dd:D,D]=l2_penalty_x2u1
            J0[:,dd:D,D+1]=l2_penalty_x2u2

            J0[:,D,0:dd]=l2_penalty_x1u1
            J0[:,D+1,0:dd]=l2_penalty_x1u2
            J0[:,D,dd:D]=l2_penalty_x2u1
            J0[:,D+1,dd:D]=l2_penalty_x2u2

            J0[:,0:dd,0:dd]=l2_penalty_A
            J0[:,0:dd,dd:D]=l2_penalty_x1x2
            J0[:,dd:D,0:dd]=l2_penalty_x2x1
            J0[:,dd:D,dd:D]=l2_penalty_A

            J0[0,0:dd,0:dd]=0
            J0[0,dd:D,dd:D]=0

            rslds.dynamics.J0 = J0

            h0 = np.concatenate((l2_penalty_A * np.eye(D),
                                    np.zeros((M + 1, D))))
            h0 = np.tile(h0[None, :, :], (K, 1, 1))

            h0[:,0:dd,0:dd]=l2_penalty_A
            h0[:,0:dd,dd:D]=l2_penalty_x1x2
            h0[:,dd:D,0:dd]=l2_penalty_x2x1
            h0[:,dd:D,dd:D]=l2_penalty_A

            h0[0,0:dd,0:dd]=0
            h0[0,dd:D,dd:D]=0

            rslds.dynamics.h0 = h0

            # model fitting ----------------------------------------------------------------
        
            # for param in range(0,4):
            Vs=np.zeros((K,D,M))
            Vs[0][0,0]=init_params[param]['v1']
            Vs[0][1,1]=init_params[param]['v1']
            Vs[0][0,1]=init_params[param]['v12']
            Vs[0][1,0]=init_params[param]['v21']
            rslds.dynamics.Vs=Vs
            rslds.dynamics.bs=np.zeros((K,D))
            Sigma=init_params[param]['sigma']*np.tile(np.eye(2),(K,1,1))
            rslds.dynamics.Sigmas=Sigma
            rslds.dynamics.Sigmas_init=rslds.dynamics.Sigmas
            A=np.tile(np.eye(2),(K,1,1))
            A[0]=init_params[param]['a1']*np.eye(2)
            rslds.dynamics.As=A
            try:
                q_elbos_ar, q_ar = rslds.fit(Y, inputs=U, method="laplace_em",
                                                                            variational_posterior="structured_meanfield", 
                                                                            continuous_optimizer='newton',
                                                                            initialize=False, 
                                                                            num_init_restarts=10,
                                                                            num_iters=50, 
                                                                            alpha=0.3, variational_posterior_kwargs={"initial_variance":1e-2})

                print("End of fitting "+str(param))
                init_A1=init_params[param]['a1']
                init_A2=init_params[param]['a2']
                init_V=init_params[param]['v1']
                init_V12=init_params[param]['v12']
                init_V21=init_params[param]['v21']
                init_sigma=init_params[param]['sigma']

                lat=latency_smpl
                if(ignore_latency):
                    lat=0

                model={'dt':dt,'init_A1':init_A1,'init_A2':init_A2,'init_V1':init_V,'init_V12':init_V12,'init_V21':init_V21,'init_sigma':init_sigma,'Y':Y , 'rslds':rslds, 'q_ar':q_ar, 'q_elbos_ar':q_elbos_ar,'inputs':U, 'K':K, "D_vec":D_vec, 'N_vec':N_vec, 'latency':lat}
                partName=name.split(".mat")
                outfile=outpath2+nameparts[0]+file_postfix[param]+'.pkl'
                with open(outfile, 'wb') as fhandle:
                    pickle.dump(model, fhandle)

            except:
                print("Error in fitting process")
