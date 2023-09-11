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

regions=['striatum']
method='compete_collaps' #single,compete,race,single_collaps
cont=np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
newpath='../dataset/Accumulator/spikes_subsample_postWheel_bestfiles/'
shuffel=50
samples=300
dt=0.001
# times=dt*np.linspace(1,samples,samples)

for r in range(len(regions)):#loop over regions
    region=regions[r]
    outpath=newpath+region

    outpath2=outpath+'/_results_'+method+'/'
    resPath=outpath2+'_timescaleData/'
    if(os.path.exists(resPath)==0):
        os.mkdir(resPath)

    onlyfiles_raw = [f for f in listdir(outpath) if isfile(join(outpath, f))]
    doneData = [f for f in listdir(resPath) if isfile(join(resPath, f))]
    # for n in range(len(onlyfiles_raw)):
    for n in range(20,len(onlyfiles_raw)):
        nameparts_raw=onlyfiles_raw[n].split(".mat")
        if(nameparts_raw[0]+'_samples.mat' in doneData):
            continue
        
        filepath=os.path.join(outpath2,nameparts_raw[0])
        if(not os.path.exists(filepath)):
            continue
        
        model_files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        if(len(model_files)==0):
            continue
            
        try:
            model = pickle.load(open(filepath+'/'+model_files[0], "rb" ))

            spikes=model['Y']
            (smpl,unitNo)=np.shape(spikes[0])
            rslds=model['rslds']
            U=model['inputs']
            
            q_ar=model['q_ar']
            spikes=model['Y']

            for s in range(len(spikes)):
                spikes[s]=np.int32(spikes[s])
                
            latency_smpl=model['latency']
            dt=model['dt']
            Y_main=model['Y']
            # medRt_smpl_totalCond=int(np.median(RT)/dt-latency_smpl)
            trialNo=len(U)

            # bootsrapping trials within each contrast level ----------------------
            totalRes=[]
            for s in range(shuffel):
                X=[]
                Z=[]
                Y=[]
                for t in range(trialNo):
                    start=np.int32(U[t][0][-1]/dt)
                    times=dt*np.linspace(start,start+samples,samples)
                    if('single' in method):
                        ev=np.ones((samples,1))*U[t][0][0]
                    else:
                        ev=np.hstack((np.ones((samples,1))*U[t][0][0], np.ones((samples,1))*U[t][0][1]))

                    u=np.hstack((ev,times[:,None]))
                    z, x, y = rslds.sample(samples, input=u)

                    Z.append(z)
                    X.append(x)
                    Y.append(y)

                mdic={'Y':Y,'Z':Z}
                totalRes.append(mdic)

        except:
            print("error in file {}".format(onlyfiles_raw[n]))

        mdic={'result':totalRes}
        outfile_mat=resPath+nameparts_raw[0]+'_samples.mat'
        savemat(outfile_mat,mdic)