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
import shutil

regions=['striatum']
method='compete_collaps' #single,compete,race,single_collaps
shuffel=100
cont=np.array([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
filepath='../dataset/Accumulator/spikes_subsample_postWheel/'
newpath='../dataset/Accumulator/spikes_subsample_postWheel_bestFit/'

if(not os.path.exists(newpath)):
    os.mkdir(newpath)

validThresh=0.03
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
color_names = [
    "purple",
    "red",
    "amber",
    "faded green",
    "windows blue",
    "orange"
]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

for r in range(len(regions)):#loop over regions
    region=regions[r]
    outpath=filepath+region

    outpath2=outpath+'/_results_'+method+'/'
    resPath=outpath2+'_BestLogLike/'
    if(os.path.exists(resPath)==0):
        os.mkdir(resPath)
    
    newOutPath=newpath+regions[r]+'/_results_'+method+'/'
    if(not os.path.exists(newOutPath)):
        os.makedirs(newOutPath)

    onlyfiles_raw = [f for f in listdir(outpath) if isfile(join(outpath, f))]

    for n in range(len(onlyfiles_raw)):    
        nameparts_raw=onlyfiles_raw[n].split(".mat")
        EV_Path=outpath+'/_results_'+method+'/'+nameparts_raw[0]+'/_single_EV/'
        
        outpath3=outpath+'/_results_'+method+'/'+nameparts_raw[0]+'/'
        newOutPath2=newOutPath+'/'+nameparts_raw[0]+'/'
        if(not os.path.exists(newOutPath2)):
            os.makedirs(newOutPath2)

        if(os.path.exists(EV_Path)==0):
            continue

        onlyfiles = [f for f in listdir(EV_Path) if isfile(join(EV_Path, f))]
        total_LL=[]
        total_LL2=[]
        name_files=[]
        fitNames=[]
        for n2 in range(len(onlyfiles)):#loop over populations
            nameparts=onlyfiles[n2].split("_EV.mat")
            
            # load log likelihood --------------------------------------
            LL_data=scipy.io.loadmat(EV_Path+onlyfiles[n2])
            fitNames.append(nameparts[0])
            total_LL.append(LL_data['logLike'][0])

        if(len(total_LL)==0):
            continue

        # find best LL -----------------------------------------------
        max_idx=total_LL.index(np.nanmax(total_LL))
        logLike=total_LL[max_idx][0]

        src_path=outpath3+fitNames[max_idx]+'.pkl'
        dst_path=newOutPath2+fitNames[max_idx]+'.pkl'
        shutil.copyfile(src_path, dst_path)
        
    