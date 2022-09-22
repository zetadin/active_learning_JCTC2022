import matplotlib
if __name__== "__main__":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time
import sys
import importlib
import os
import hashlib
import subprocess
import tempfile
import gc
import queue
import threading

import torch

from matplotlib import gridspec
from datetime import datetime
from numpy.random import choice
from tqdm import tqdm, trange
from copy import deepcopy
from sklearn.cluster import KMeans

try:
    import cPickle as pickle
except:
    import pickle

energy_folder="/home/energy/ykhalak/ML_dG/pde2_dG/retrospective_active_learning_in_6EZF/"
energy_folder_workstation="/netmount/energy/ykhalak/ML_dG/pde2_dG/retrospective_active_learning_in_6EZF/"
all_ligs_db_file_energy=f"{energy_folder_workstation}/full_ligs.pickle"


all_ligs_db_file_old="/home/ykhalak/Projects/ML_dG/pde2_dG/how_do_ligs_fit_in_pocket/adaptive_learning_test_from_morphed_structs/../processed_ligs_w_morphing_sim_annealing_only_sucessfull.pickle"
experiment_folder="/home/ykhalak/Projects/ML_dG/pde2_dG/retrospective_active_learning_in_6EZF/"
all_ligs_db_file=f"{experiment_folder}/full_ligs.pickle"

datafolder=f"{experiment_folder}/cached_reprs"


Bohr2Ang=0.529177249
RT=0.001985875*300 #kcal/mol

import sys
#sys.path.append(f"{script_folder}/..")
#sys.path.append(script_folder)

#import utils
from utils import *

#build NN class
import NNs
from NNs import *

#build data loader
import custom_dataset_modular_with_binning_6EZF
from custom_dataset_modular_with_binning_6EZF import dataBlocks, CustomMolModularDataset


flags_2D=[dataBlocks.MACCS, dataBlocks.Descriptors, dataBlocks.Graph_desc, dataBlocks.BCUT2D]
flags_3D=[dataBlocks.EState_FP, dataBlocks.Pharmacophore_feature_map,
        dataBlocks.MOE, dataBlocks.MQN, dataBlocks.GETAWAY, dataBlocks.AUTOCORR2D,
        dataBlocks.AUTOCORR3D, dataBlocks.WHIM, dataBlocks.RDF,
        dataBlocks.USR, dataBlocks.USRCUT, dataBlocks.PEOE_VSA, dataBlocks.SMR_VSA,
        dataBlocks.SlogP_VSA, dataBlocks.MORSE]

flags_2D_3D=flags_2D+flags_3D
    
    
class LearningHistory:
    def __init__(self):
        self.metrics_auto=["RMSD", "Cor", "TP", "FP", "AUC"]
        self.metrics=["RMSD", "Cor", "TP", "FP", "AUC", "prec", "KT"]
        self.datasubsets=["XVal", "selected", "unmeasured", "top10", "top50", "top244"]
        
        self.top10_found=[]
        self.top50_found=[]
        self.top244_found=[]
        
        self.known_lig_ids=[]
        self.step_lig_ids=[]
        
        for m in self.metrics:
            for d in self.datasubsets:
                setattr(self, m+'_'+d, [])


from enum import Enum, auto
class SelectionRule(Enum):
    
    greedy = 0
    narrowing = auto()
    uncertain = auto()
    schroedinger = auto()
    weighted_Tanimoto_random = auto()
    weighted_2D_repr_random = auto()
    random = auto()
    random2greedy = auto()
    
    def __int__(self):
        return self.value
    
class StartingSelectionMethod(Enum):
    
    random = 0
    weighted_by_2D_repr = auto()
    weighted_by_RDKFP_Tanimoto = auto()
    from_two_Tanimoto_clusters = auto()
    from_Tanimoto_clusters2 = auto()
    from_Tanimoto_clusters3 = auto()
    from_Tanimoto_clusters4 = auto()
    
    def __int__(self):
        return self.value
    
################################################################################
################################Helper functions################################
################################################################################

def select_starting_ligs(starting_method, n_ligs, n_picks=100, seed=None, used_lig_idxs=None):
    
    if(seed is not None):
        np.random.seed(seed)
    if(used_lig_idxs is None):
        used_lig_idxs=np.arange(n_ligs)
    
    indeces=np.arange(n_ligs)
    if(starting_method==StartingSelectionMethod.random):
        return(np.random.choice(indeces, n_picks, replace=False))
    elif(starting_method==StartingSelectionMethod.weighted_by_2D_repr or
         starting_method==StartingSelectionMethod.weighted_by_RDKFP_Tanimoto):
        if(starting_method==StartingSelectionMethod.weighted_by_2D_repr):
            ebedding_fn=experiment_folder+"/2D_repr_embedded_by_tSNE.pickle"
        elif(starting_method==StartingSelectionMethod.weighted_by_RDKFP_Tanimoto):
            ebedding_fn=experiment_folder+"/RDFFP_repr_Tanimoto_embedded_by_tSNE.pickle"
        else:
            raise()
        with open(ebedding_fn, 'rb') as f:
            S2 = pickle.load(f)
            S2 = S2[used_lig_idxs,:]
            
        lims2=[(-100,100),(-100,100)]
        bins=[200,200]

        counts2,edges2=np.histogramdd(S2, bins=bins, range=lims2, density=False)

        starts2=np.array([edges2[i][0] for i in range(2)])
        steps2=np.array([(edges2[i][-1]-edges2[i][0])/(len(edges2[i])-1) for i in range(2)])

        inds2=np.floor((S2-starts2[np.newaxis,:])/steps2[np.newaxis,:]).astype(int)

        probabilities2=1/counts2[inds2[:,0],inds2[:,1]]
        probabilities2/=np.sum(probabilities2)
        #print(f"pick {n_picks} from {probabilities2.shape} or {indeces.shape}")
        return(np.random.choice(indeces, n_picks, p=probabilities2, replace=False))
            
    elif(starting_method==StartingSelectionMethod.from_two_Tanimoto_clusters):
        ebedding_fn=experiment_folder+"/RDFFP_repr_Tanimoto_embedded_by_tSNE.pickle"
        with open(ebedding_fn, 'rb') as f:
            S2 = pickle.load(f)
            S2 = S2[used_lig_idxs,:]
        center=np.array([-60,-15])
        disp=S2-center[np.newaxis,:]
        distsq=np.sum(disp*disp,axis=1)
        mask=distsq<25*16
        candidates=np.argwhere(mask).flatten()
        return(np.random.choice(candidates, n_picks, replace=False))
        
    elif(starting_method==StartingSelectionMethod.from_Tanimoto_clusters2):
        ebedding_fn=experiment_folder+"/RDFFP_repr_Tanimoto_embedded_by_tSNE.pickle"
        with open(ebedding_fn, 'rb') as f:
            S2 = pickle.load(f)
            S2 = S2[used_lig_idxs,:]
        center1=np.array([-30,65])
        disp1=S2-center1[np.newaxis,:]
        distsq1=np.sum(disp1*disp1,axis=1)
        center2=np.array([0,75])
        disp2=S2-center2[np.newaxis,:]
        distsq2=np.sum(disp2*disp2,axis=1)
        mask=np.logical_or(distsq1<25*16, distsq2<25*16)
        candidates=np.argwhere(mask).flatten()
        return(np.random.choice(candidates, n_picks, replace=False))
        
    elif(starting_method==StartingSelectionMethod.from_Tanimoto_clusters3):
        ebedding_fn=experiment_folder+"/RDFFP_repr_Tanimoto_embedded_by_tSNE.pickle"
        with open(ebedding_fn, 'rb') as f:
            S2 = pickle.load(f)
            S2 = S2[used_lig_idxs,:]
        center1=np.array([46,43])
        disp1=S2-center1[np.newaxis,:]
        distsq1=np.sum(disp1*disp1,axis=1)
        mask=distsq1<25*20
        candidates=np.argwhere(mask).flatten()
        return(np.random.choice(candidates, n_picks, replace=False))
        
    elif(starting_method==StartingSelectionMethod.from_Tanimoto_clusters4):
        ebedding_fn=experiment_folder+"/RDFFP_repr_Tanimoto_embedded_by_tSNE.pickle"
        with open(ebedding_fn, 'rb') as f:
            S2 = pickle.load(f)
            S2 = S2[used_lig_idxs,:]
        center1=np.array([-5,-48])
        disp1=S2-center1[np.newaxis,:]
        distsq1=np.sum(disp1*disp1,axis=1)
        mask=distsq1<25*4
        candidates=np.argwhere(mask).flatten()
        return(np.random.choice(candidates, n_picks, replace=False))
        
    else:
        raise(Exception("Unrecognized method for selecting starting ligands"))

################################################################################

def select_ligs_weighted(candiadate_inds, method, n_picks=100):
        if(method==SelectionRule.weighted_2D_repr_random):
            ebedding_fn=experiment_folder+"/2D_repr_embedded_by_tSNE.pickle"
        elif(method==SelectionRule.weighted_Tanimoto_random):
            ebedding_fn=experiment_folder+"/RDFFP_repr_Tanimoto_embedded_by_tSNE.pickle"
        else:
            raise(Exception("Unsupported method for randomly selecting ligands"))
        
        with open(ebedding_fn, 'rb') as f:
            S2 = pickle.load(f)
            
        lims2=[(-100,100),(-100,100)]
        bins=[200,200]

        counts2,edges2=np.histogramdd(S2, bins=bins, range=lims2, density=False)

        starts2=np.array([edges2[i][0] for i in range(2)])
        steps2=np.array([(edges2[i][-1]-edges2[i][0])/(len(edges2[i])-1) for i in range(2)])

        inds2=np.floor((S2-starts2[np.newaxis,:])/steps2[np.newaxis,:]).astype(int)

        probabilities2=1/counts2[inds2[:,0],inds2[:,1]]
        probabilities2=probabilities2[candiadate_inds]
        probabilities2/=np.sum(probabilities2)
        
        return(np.random.choice(candiadate_inds, n_picks, p=probabilities2, replace=False))
    

################################################################################

def get_represetations():    
    representations=[]
    representation_names=[]

    block_names_full=["2D_3D", "PLEC_filtered", "MDenerg", "MDenerg_long_cut", "MDenerg_binned", "MDenerg_long_cut_binned", "atom_hot", "atom_hot_surf"]

    combs=np.eye(len(block_names_full), dtype=bool)

    for c in combs:
        b_2D_3D, b_PLEC_filt, b_MDenerg, b_MDenerg_lc, b_MDenerg_binned, b_MDenerg_lc_binned, b_ah, b_ah_vdw = c
        representation_flags=[0]*len(dataBlocks)
        
        if(b_2D_3D):
            for b in flags_2D:
                representation_flags[int(b)]=1
            for b in flags_3D:
                representation_flags[int(b)]=1

        if(b_PLEC_filt):
            representation_flags[int(dataBlocks.PLEC_filtered)]=1

        if(b_MDenerg):
            representation_flags[int(dataBlocks.MDenerg)]=1

        if(b_MDenerg_lc):
            representation_flags[int(dataBlocks.MDenerg_longcut)]=1

        if(b_MDenerg_binned):
            representation_flags[int(dataBlocks.MDenerg_binned)]=1

        if(b_MDenerg_lc_binned):
            representation_flags[int(dataBlocks.MDenerg_longcut_binned)]=1

        if(b_ah):
            representation_flags[int(dataBlocks.atom_hot)]=1

        if(b_ah_vdw):
            representation_flags[int(dataBlocks.atom_hot_on_vdw_surf)]=1


        dr_name=""
        for i in range(len(c)):
            if c[i]:
                dr_name+=block_names_full[i]+'_'
        dr_name=dr_name[:-1]
        
        
        representations.append(representation_flags)
        representation_names.append(dr_name)
        
    return(representations, representation_names)


################################################################################

def launch_moa_trainers(sfiles, AL_settings_hash, lig_db_fn, step, lig_ids, max_workers=10, use_moa=True):
    
    lig_ids_simple=""
    for lid in lig_ids:
        lig_ids_simple+=f"{lid} "
    
    previously_done=0
    max_debug_evals=10e8

    q = queue.Queue()
    for settings_fname in sfiles:
        with open(settings_fname, 'rb') as f:
            settings_loaded, metrics = pickle.load(f)
            if(metrics is None):
                q.put(settings_fname)
                print(settings_fname)
                if(q.qsize()>=max_debug_evals):
                    break;
            else:
                previously_done+=1

    print("previously_done:", previously_done, "\t out of:", len(sfiles), flush=True)
    nleft=len(sfiles)-previously_done
    nworkers=max(min(max_workers, nleft), 1)
    n_per_worker=int(np.ceil(float(nleft)/nworkers))
    print("settings left:", nleft, "\t # workers:", nworkers, "\t # settings/worker:", n_per_worker)

    if(nleft==0):
        return([])
    #raise()

    cwd=f"{experiment_folder}/AL_settings/run_{AL_settings_hash}/step_{step}/jobscripts"
    cmd_str=f"source /etc/profile; module load sge; cd {cwd};"
    cmd_str_SLURM=f"source /etc/profile; cd {cwd};"
    
    save_folder_base=f"{experiment_folder}/AL_settings/run_{AL_settings_hash}/step_{step}"

    def worker(job_id):
        sf_str=""
        for l in range(n_per_worker):
            fname = q.get()
            if fname is None:  # EOF?
                break
            sf_str+=" "+fname
        os.makedirs(cwd, exist_ok=True)
        jobscript_str=f"""#!/bin/bash
        
#$ -S /bin/bash
#$ -pe openmp_fast 1
#$ -q *
#$ -N AL_{AL_settings_hash}_step_{step}_{job_id}
#$ -M ykhalak@gwdg.de
#$ -m n
#$ -l h_rt=3:00:00
#$ -wd {cwd}

#SBATCH --job-name=pytorch_SLURM_GPU_worker_{job_id}
#SBATCH --get-user-env
#SBATCH --gres=gpu:1              # number of GPUs requested
#SBATCH --ntasks=1                # Number of MPI process
####SBATCH --cpus-per-task=6         # CPU cores per MPI process
#SBATCH -p p24,p20,p16,p10,p08,p06    # partitions to use
#SBATCH -t 3:00:0                 # hours:min:sec
#SBATCH --chdir={cwd}
#SBATCH -e AL_{AL_settings_hash}_step_{step}_{job_id}-%j.err
#SBATCH -o AL_{AL_settings_hash}_step_{step}_{job_id}-%j.out

if [ -n "$NHOSTS" ]; then
    echo "Env. variable NHOSTS found, assuming we are using SGE."
    #module load shared                           # access to modules in /cm/shared
else
    echo "Env. variable NHOSTS not set, assuming we are using SLURM."
    export NHOSTS=$SLURM_JOB_NUM_NODES
    #export NSLOTS=$SLURM_CPUS_PER_TASK
    export NSLOTS=$SLURM_CPUS_ON_NODE 
fi

export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

source ~/.ML_v2_profile

cp /home/ykhalak/Projects/ML_dG/pde2_dG/retrospective_active_learning_in_6EZF/cached_reprs/modular_repr_cache.tar.gz $TMPDIR/.
cd $TMPDIR
tar -zxf modular_repr_cache.tar.gz
cd {cwd}

python /home/ykhalak/Projects/ML_dG/pde2_dG/retrospective_active_learning_in_6EZF/owl_trainer_w_params_general_with_ensemble_summary_model_6EZF.py -v -f {sf_str} --ligsf {lig_db_fn} --datafolder $TMPDIR --save_folder_base {save_folder_base} --training_lig_ids {lig_ids_simple} #--sm

    """
        jobscript_fn=cwd+"/jobscript_{}".format(job_id)
        with open(jobscript_fn,"w") as f:
            f.write(jobscript_str)

        #global cmd_str, cmd_str_SLURM
        nonlocal cmd_str, cmd_str_SLURM
        cmd_str+=" qsub {};".format(jobscript_fn)
        cmd_str_SLURM+=" sbatch {};".format(jobscript_fn)

    for job_id in range(nworkers):
        q.put(None)
        worker(job_id)
        
    #raise()

    print("Submitting.")
    if(use_moa):
        ssh_cmd_arr=["ssh", "moa1", cmd_str_SLURM]
    else:
        ssh_cmd_arr=["ssh", "owl", cmd_str]
    process = subprocess.Popen(ssh_cmd_arr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    outs, _ = process.communicate()
    process.wait()
    #print('== subprocess exited with rc =', process.returncode)
    print(outs.decode('utf-8'))
    #print("Done.")
    
    if("submission failed" in outs.decode('utf-8')):
        raise(Exception("Failed to submit jobs to cluster."))
    
    
    #read cluster_job_ids that were just launched
    cluster_job_ids=[]
    if(use_moa):
        for l in outs.decode('utf-8').splitlines():
            if("Submitted batch job" in l):
                cluster_job_ids.append(int(l.split()[-1]))
    else:
        for l in outs.decode('utf-8').splitlines():
            if("Your job" in l and "has been submitted" in l):
                cluster_job_ids.append(int(l.split()[2]))
    
    return(cluster_job_ids)

################################################################################

def rank_representations(sfiles, verbose=False):
    traning_RMSE={}
    XVals_RMSE={}
    XVals_Cor={}
    XVals_KT={}
    XVals_TPR={}

    for fn in sfiles:
        with open(fn, 'rb') as f:
            metrics=None
            try:
                settings_loaded, metrics = pickle.load(f)
            except Exception:
                print(f"problem with {fn}")
            if(metrics is not None):
                key=f"{settings_loaded[1]}_hlw{settings_loaded[5]}"
                bi=np.argmin(metrics.loss_XVal)

                XVals_RMSE[key]=metrics.RMSD_XVal[bi]
                traning_RMSE[key]=metrics.RMSD_Train[bi]
                XVals_Cor[key]=metrics.Cor_XVal[bi]
                XVals_KT[key]=metrics.kendalltau_XVal[bi]
                XVals_TPR[key]=metrics.TP_XVal[bi]

    sorted_RMSE=dict(sorted(XVals_RMSE.items(), key=lambda item: item[1]))
    sorted_Cor= dict(sorted(XVals_Cor.items(), key=lambda item: item[1]))
    sorted_KT=  dict(sorted(XVals_KT.items(), key=lambda item: item[1]))
    sorted_TPR= dict(sorted(XVals_TPR.items(), key=lambda item: item[1]))
    
    nranks=5
    def rank_by_kmeans(d, nranks=nranks):
        X = np.array([[v[1]] for v in d.items()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=nranks, random_state=0).fit(X)
        sorted_cl_ids=np.argsort(kmeans.cluster_centers_.flatten())
        rank_from_cl_id={}
        for i,j in enumerate(sorted_cl_ids):
            rank_from_cl_id[j]=i
        
        ret={}
        for i,key in enumerate(d.keys()):
            ret[key]=rank_from_cl_id[kmeans.labels_[i]]
        return(ret)

    rank_RMSE=rank_by_kmeans(sorted_RMSE)
    rank_Cor=rank_by_kmeans(sorted_Cor)
    rank_KT=rank_by_kmeans(sorted_KT)
    rank_TPR=rank_by_kmeans(sorted_TPR)

    rank={}
    for key in rank_RMSE.keys():
        rank[key]=rank_RMSE[key]+(nranks-1-rank_Cor[key])+(nranks-1-rank_KT[key])+(nranks-1-rank_TPR[key])
        
    sorted_rank= dict(sorted(rank.items(), key=lambda item: item[1]))
    keys=list(sorted_rank.keys())

    if(verbose):
        print(f"{'RANK':^4}\t{'REPRESENTATION':^40}\t{'XVal RMSE':^10}\t{'XVal Cor':^10}\t{'XVal Ken.-t':^10}\t{'XVal TPR':^10}")
        for i in range(len(sorted_RMSE)):
            key=keys[i]
            r=rank[key]
            print(f"{r:4}\t{key:40}\t{XVals_RMSE[key]:.4f}\t\t{XVals_Cor[key]:.4f}\t\t{XVals_KT[key]:.4f}\t\t{XVals_TPR[key]:.4f}")
            
    return(keys[:5])

    
################################################################################

def AL_Trainer(settings_fn, lig_db_fn, redo=False, verbose=False, experiment_folder_override=experiment_folder, use_moa=True, iter_through_full_DSet=False, save_Preds=False):
    global experiment_folder
    experiment_folder=experiment_folder_override
    
    #0. load the settings
    with open(settings_fn, 'rb') as f:
        learner_settings, learner_metrics = pickle.load(f)
        write_AL_metrics_toHDD=False
    if(learner_metrics is not None and not redo):
        print(f"learner_settings has data for {len(learner_metrics.top10_found)} steps")
        if(len(learner_metrics.top10_found)>=6):
            print("already handled previously")
            return()
    else:
        learner_metrics=LearningHistory()
        write_AL_metrics_toHDD=True
    
    normalize_x, shuffle_seed, n_Epochs, init_learning_rate, learning_rate_decay, weight_decay,\
        impfilt, X_filter, weighted, shiftY, use_dropout, sele_rule, starting_method,\
        hlw, hld, starting_ligs_seed, n_picks_per_step, fixed_repr, select_on_summary_model,\
        fraction_of_actives_to_use = learner_settings[:20] # for now only the parameters actually used
        
    if(fraction_of_actives_to_use is None):
        fraction_of_actives_to_use=1.0
        
    print(f"requested selection rule: {sele_rule}")
    
    if(fixed_repr is not None and sele_rule==SelectionRule.narrowing):
        raise(Exception("SelectionRule.narrowing is inpompatible with a fixed representation."))
    if(fixed_repr is not None and len(fixed_repr)!=2):
        raise(Exception("fixed_repr should be a tupple: (representation_flags, name_string)."))

    if(select_on_summary_model is None):
        select_on_summary_model=False
        
    #1. find the names and paths for settigns and results
    sha = hashlib.sha256()
    sha.update(pickle.dumps(learner_settings))
    AL_settings_hash=sha.hexdigest()[:6]
    
    settings_folder=f"{experiment_folder}/AL_settings/run_{AL_settings_hash}"
    #os.makedirs(settings_folder, exist_ok=True)
        
        
    #2. load the ligands
    custom_lig_db_fn=lig_db_fn
    localized_lig_db_fn=lig_db_fn
    if(localized_lig_db_fn[:13]=="/home/energy/"):
        localized_lig_db_fn="/netmount/energy/"+localized_lig_db_fn[13:]
    with open(localized_lig_db_fn, 'rb') as f:
        ligs = pickle.load(f)
        
        if(fraction_of_actives_to_use<1.0):
            # parse through the ligands and filter out the ones we don't want in this test
            dG_all=np.array([float(lig.GetProp('dG')) for lig in ligs])
            soted_lig_ids=np.argsort(dG_all)
            active_cutoff=-11
            actives=np.where(dG_all<active_cutoff)[0]
            inactives=np.where(dG_all>active_cutoff)[0]
            np.random.seed(starting_ligs_seed*4759+1) # a seed that is different to the one used in selecting first training ligands, but related to it.
            used_actives=np.random.choice(actives, int(np.rint(fraction_of_actives_to_use*len(actives))), replace=False)
            used_ids=np.concatenate((used_actives,inactives))
            ligs=[ligs[i] for i in used_ids] # new filtered ligands
            # dump this new sub-library into a pickle file and point the owl_trainer at it
            custom_lig_db_fn=f"{experiment_folder}/AL_settings/run_{AL_settings_hash}/sublibrary.pickle"
            pickle.dump( ligs, open(custom_lig_db_fn, "wb" ) )
            if(verbose):
                print(f"Retaining {len(used_actives)}/{len(actives)} of active ligands")
                
        else:
            used_ids=np.arange(len(ligs))

    
    n_ligs=len(ligs)
    dG_all=np.array([float(lig.GetProp('dG')) for lig in ligs])
    soted_lig_ids=np.argsort(dG_all)
    top10_ids = soted_lig_ids[:10]
    top50_ids = soted_lig_ids[:50]
    top244_ids= soted_lig_ids[:244]
    if(verbose):
        print("Number of ligands:", n_ligs)
    
    
    #3. select starting ligands
    step_0_lig_ids = select_starting_ligs(starting_method, n_ligs=n_ligs, n_picks=n_picks_per_step, seed=starting_ligs_seed, used_lig_idxs=used_ids)
    step_0_lig_ids = np.array(step_0_lig_ids, dtype=np.uint)
    del ligs
    _=gc.collect()
    
    step_lig_ids=[step_0_lig_ids]
    known_lig_ids=step_0_lig_ids
    
    #4. get all the relevant representations
    possible_reprs, possible_repr_names = get_represetations()
    
    #how many iterations should we go through
    if(iter_through_full_DSet):
        n_iter=int(np.floor(n_ligs/n_picks_per_step))
    else:
        n_iter=7
    
    #5. loop over AL iterations
    for step in range(n_iter):
        if(verbose):
            print("\n\nStarting step", step)
        #step_ligs=ligs[step_lig_ids[-1]]
        
        need_training=True
        #if(sele_rule==SelectionRule.random or sele_rule==SelectionRule.weighted_Tanimoto_random or sele_rule==SelectionRule.weighted_2D_repr_random):
            #need_training=False
        
        #5.1 choose representations to train
        if(sele_rule==SelectionRule.narrowing):
            if(step<3):
                target_reprs=possible_reprs
                target_repr_names=possible_repr_names            
            else:
                repr_2D_3D_flags=[0]*len(dataBlocks)
                for b in flags_2D:
                    repr_2D_3D_flags[int(b)]=1
                for b in flags_3D:
                    repr_2D_3D_flags[int(b)]=1
                target_reprs=[repr_2D_3D_flags]
                target_repr_names=["2D_3D"]
        else:
            repr_2D_3D_flags=[0]*len(dataBlocks)
            for b in flags_2D:
                repr_2D_3D_flags[int(b)]=1
            for b in flags_3D:
                repr_2D_3D_flags[int(b)]=1
            target_reprs=[repr_2D_3D_flags]
            target_repr_names=["2D_3D"]
            
        #override representation
        if(fixed_repr is not None):
            target_reprs=[fixed_repr[0]]
            target_repr_names=[fixed_repr[1]]
        
        
        if(need_training):
            #5.2 choose number of repeats to use in ensemble model
            if(sele_rule==SelectionRule.narrowing):
                if(step<3):
                    n_repeats=1
                else:
                    n_repeats=5
            else:
                n_repeats=5
                
            #5.3 write the settings files
            step_folder=f"{settings_folder}/step_{step}"
            step_settings_folder=f"{step_folder}/settings/"
            os.makedirs(step_settings_folder, exist_ok=True)
            
            if(hlw=='auto' or hld=='auto' or n_Epochs=='auto'):
                if(step<3):
                    hlw_use=10
                    hld_use=2
                else: # after the metaparameter optimizations
                    hlw_use=20
                    hld_use=3
                
                if(step==0):
                    n_Epochs_use=2000
                    hlw_use=300 # overwrite width of first step
                else:
                    n_Epochs_use=20000
            else:
                hlw_use=hlw
                hld_use=hld
                n_Epochs_use=n_Epochs
            
            sfiles=[]
            for ir, repres in enumerate(target_reprs):
                for repeat in range(n_repeats):
                    dr_name=f"{target_repr_names[ir]}_repeat{repeat}"
                    settings=[
                        repres, dr_name,
                        normalize_x, shuffle_seed,
                        n_Epochs_use, hlw_use, hld_use,
                        init_learning_rate, learning_rate_decay, weight_decay,
                        impfilt, X_filter,
                        weighted, shiftY, use_dropout
                        ]
                    
                    sha = hashlib.sha256()
                    sha.update(pickle.dumps(settings))
                    settings_hash=dr_name+"_"+sha.hexdigest()[:10]
                    
                    settings_fname=f"{step_settings_folder}/{settings_hash}.pickle"
                    if(os.path.exists(settings_fname)):
                        sfiles.append(settings_fname)
                        #with open(settings_fname, 'rb') as f:
                            #settings_loaded, metrics = pickle.load(f)
                            #if(metrics is not None and not redo):
                                #print(f"{settings_fname} is already trained.")
                            #else:
                                #print(f"{settings_fname} exists but is not trained yet.")
                                #sfiles.append(settings_fname)
                    else:
                        pickle.dump( (settings, None), open( settings_fname, "wb" ) )
                        sfiles.append(settings_fname)
                        
            #print(sfiles)
            #raise() # debug stop before bothering moa
                        
            #5.4 submit training to moa
            cluster_job_ids=launch_moa_trainers(sfiles, AL_settings_hash, custom_lig_db_fn, step, known_lig_ids, max_workers=10, use_moa=use_moa)
            
            #5.5 wait for trainers to finish
            ready=False
            if(len(cluster_job_ids)==0):
                ready=True
            wait_start=time.time()
            while not ready:
                time.sleep(30) # 0.5 min
                ready=[os.path.getsize(sf)>500 for sf in sfiles]
                #if(verbose):
                    #print("checking settings files for completition:", ready)
                ready=all(ready)
                
                if(not ready):
                    now=time.time()
                    elapsed=now-wait_start
                    if(elapsed>3*60): # 3 min
                        # if we have been waiting for more than 10 minutes and job hasn't finished, check the cluster queue to see if jobs have crashed
                        if(use_moa):
                            if(verbose):
                                print("Polling moa for status of running jobs.")
                            ssh_cmd_arr=["ssh", "moa1", "squeue -u ykhalak"]
                            process = subprocess.Popen(ssh_cmd_arr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                            outs, _ = process.communicate()
                            process.wait()
                            outs=outs.decode('utf-8')
                        else:
                            if(verbose):
                                print("Polling owl for status of running jobs.")
                            ssh_cmd_arr=["ssh", "owl", "module use --append /cm/shared/modulefiles; module load sge; qstat"]
                            process = subprocess.Popen(ssh_cmd_arr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                            outs, _ = process.communicate()
                            process.wait()
                            outs=outs.decode('utf-8')
                        
                        jobs_in_queue=[str(cjid) in outs for cjid in cluster_job_ids]
                        
                        if(verbose):
                            for j in range(len(cluster_job_ids)):
                                print(cluster_job_ids[j], jobs_in_queue[j])
                            
                        if(not any(jobs_in_queue)):
                            # all jobs are done, but some outputs are missing
                            if(not all([os.path.getsize(sf)>500 for sf in sfiles])): #double check readiness
                                raise(Exception(f"all jobs are done, but some outputs are missing: {step_folder}"))
                            else:
                                ready=True
                        else:
                            # reset wait timer
                            wait_start=time.time()
            
            #time.sleep(5) # 5 sec to make sure I/O finishes
            #raise()
        
        
        
        
        #unmeasured_ids=np.arange(n_ligs)[~known_lig_ids]
        unmeasured_ids=np.array([i for i in range(n_ligs) if i not in np.unique(known_lig_ids)])
        
        if(sele_rule==SelectionRule.narrowing):
            if(step<3):
                #5.6.a Select 5 best representations
                if(verbose):
                    print("Ranking representations")
                best_reprs=rank_representations(sfiles, verbose)
            
                #5.6.b Select ligands for next step
                if(verbose):
                    print("Selecting next ligands")
                
                new_lig_ids=[]
                for fn in sfiles:
                    with open(fn, 'rb') as f:
                        metrics=None
                        try:
                            settings_loaded, metrics = pickle.load(f)
                        except Exception:
                            print(f"problem with {fn}")
                        if(metrics is not None):
                            key=f"{settings_loaded[1]}_hlw{settings_loaded[5]}"
                            if(key in best_reprs):
                                #print(key)
                                if(select_on_summary_model):
                                    P_all=metrics.summary_model_best_pred[0][unmeasured_ids]
                                else:
                                    P_all=metrics.best_pred[0]
                                unmeasured_P = P_all[unmeasured_ids]
                                unmeasured_uncert = metrics.best_pred[1][unmeasured_ids]
                                
                                n_ligs_from_repr = np.ceil(n_picks_per_step/5)
                                
                                sorted_unmeasured_ids=unmeasured_ids[np.argsort(unmeasured_P)]
                                selected=0
                                n=0
                                attempted_duplicates=0
                                attempted_duplicates_by_repr=[0 for i in range(5)]
                                while selected<n_ligs_from_repr and n<len(sorted_unmeasured_ids):
                                    if(sorted_unmeasured_ids[n] not in new_lig_ids):
                                        new_lig_ids.append(sorted_unmeasured_ids[n])
                                        selected+=1
                                    else:
                                        attempted_duplicates+=1
                                        prev_repr=int(np.floor(new_lig_ids.index(sorted_unmeasured_ids[n])/n_ligs_from_repr))
                                        attempted_duplicates_by_repr[prev_repr]+=1
                                    n+=1
                                #print(len(new_lig_ids), attempted_duplicates, attempted_duplicates_by_repr)
                                
            else:
                
                if(verbose):
                    #5.6.a rank representation so model quality is in the log file
                    print("Ranking representations")
                    best_reprs=rank_representations(sfiles, verbose)
                
                metrics=None
                #average over all the repeats of 2D_3D
                P_over_repeats=[]
                for sf in sfiles:
                    with open(sf, 'rb') as f:
                        try:
                            settings_loaded, metrics = pickle.load(f)
                        except Exception:
                            print(f"problem with {fn}")
                    P_over_repeats.append(metrics.best_pred[0])
                    if(select_on_summary_model):
                        P_over_repeats.append(metrics.summary_model_best_pred[0])
                    else:
                        P_over_repeats.append(metrics.best_pred[0])
                P_avg=np.mean(P_over_repeats, axis=0)
                P_err=np.std(P_over_repeats, axis=0)#/sqrt(len(sfiles))
                unmeasured_P = P_avg[unmeasured_ids]
                unmeasured_uncert = P_err[unmeasured_ids]
                sorted_unmeasured_ids=unmeasured_ids[np.argsort(unmeasured_P)]
                new_lig_ids=sorted_unmeasured_ids[:n_picks_per_step]
                
        elif(sele_rule==SelectionRule.greedy):
            if(verbose):
                #5.6.a rank representation so model quality is in the log file
                print("Ranking representations")
                best_reprs=rank_representations(sfiles, verbose)
                
            metrics=None
            #average over all the repeats of 2D_3D
            P_over_repeats=[]
            for sf in sfiles:
                with open(sf, 'rb') as f:
                    try:
                        settings_loaded, metrics = pickle.load(f)
                    except Exception:
                        print(f"problem with {fn}")
                if(select_on_summary_model):
                    P_over_repeats.append(metrics.summary_model_best_pred[0])
                else:
                    P_over_repeats.append(metrics.best_pred[0])
            P_avg=np.mean(P_over_repeats, axis=0)
            P_err=np.std(P_over_repeats, axis=0)#/sqrt(len(sfiles))
            unmeasured_P = P_avg[unmeasured_ids]
            unmeasured_uncert = P_err[unmeasured_ids]
            sorted_unmeasured_ids=unmeasured_ids[np.argsort(unmeasured_P)]
            new_lig_ids=sorted_unmeasured_ids[:n_picks_per_step]
            
        elif(sele_rule==SelectionRule.uncertain):
            if(verbose):
                #5.6.a rank representation so model quality is in the log file
                print("Ranking representations")
                best_reprs=rank_representations(sfiles, verbose)
                
            metrics=None
            #average over all the repeats of 2D_3D
            P_over_repeats=[]
            for sf in sfiles:
                with open(sf, 'rb') as f:
                    try:
                        settings_loaded, metrics = pickle.load(f)
                    except Exception:
                        print(f"problem with {fn}")
                if(select_on_summary_model):
                    P_over_repeats.append(metrics.summary_model_best_pred[0])
                else:
                    P_over_repeats.append(metrics.best_pred[0])
            P_avg=np.mean(P_over_repeats, axis=0)
            P_err=np.std(P_over_repeats, axis=0)#/sqrt(len(sfiles))
            unmeasured_P = P_avg[unmeasured_ids]
            unmeasured_uncert = P_err[unmeasured_ids]
            sorted_unmeasured_ids=unmeasured_ids[np.argsort(unmeasured_uncert)]
            new_lig_ids=sorted_unmeasured_ids[-n_picks_per_step:]
            
        elif(sele_rule==SelectionRule.schroedinger):
            if(verbose):
                #5.6.a rank representation so model quality is in the log file
                print("Ranking representations")
                best_reprs=rank_representations(sfiles, verbose)
                
            metrics=None
            #average over all the repeats of 2D_3D
            P_over_repeats=[]
            for sf in sfiles:
                with open(sf, 'rb') as f:
                    try:
                        settings_loaded, metrics = pickle.load(f)
                    except Exception:
                        print(f"problem with {fn}")
                if(select_on_summary_model):
                    P_over_repeats.append(metrics.summary_model_best_pred[0])
                else:
                    P_over_repeats.append(metrics.best_pred[0])
            P_avg=np.mean(P_over_repeats, axis=0)
            P_err=np.std(P_over_repeats, axis=0)#/sqrt(len(sfiles))
            unmeasured_P = P_avg[unmeasured_ids]
            unmeasured_uncert = P_err[unmeasured_ids]
            sorted_unmeasured_ids=unmeasured_ids[np.argsort(unmeasured_P)]
            candidate_ids=sorted_unmeasured_ids[:3*n_picks_per_step] # select a broader range of predicted top binders
            candidate_uncert = P_err[candidate_ids]
            sorted_candidate_ids=candidate_ids[np.argsort(candidate_uncert)]
            new_lig_ids=sorted_candidate_ids[-n_picks_per_step:] # least certain from the candidates
            
        elif(sele_rule==SelectionRule.weighted_Tanimoto_random or sele_rule==SelectionRule.weighted_2D_repr_random):
            new_lig_ids=select_ligs_weighted(unmeasured_ids, sele_rule, n_picks=n_picks_per_step)
            
        elif(sele_rule==SelectionRule.random):
            new_lig_ids=np.random.choice(unmeasured_ids, n_picks_per_step, replace=False)
            
        elif(sele_rule==SelectionRule.random2greedy):
            if(verbose):
                print("Ranking representations")
                best_reprs=rank_representations(sfiles, verbose)
                
            if(step<3):
                new_lig_ids=np.random.choice(unmeasured_ids, n_picks_per_step, replace=False)
            else:                   
                metrics=None
                #average over all the repeats of 2D_3D
                P_over_repeats=[]
                for sf in sfiles:
                    with open(sf, 'rb') as f:
                        try:
                            settings_loaded, metrics = pickle.load(f)
                        except Exception:
                            print(f"problem with {fn}")
                    if(select_on_summary_model):
                        P_over_repeats.append(metrics.summary_model_best_pred[0])
                    else:
                        P_over_repeats.append(metrics.best_pred[0])
                P_avg=np.mean(P_over_repeats, axis=0)
                P_err=np.std(P_over_repeats, axis=0)#/sqrt(len(sfiles))
                unmeasured_P = P_avg[unmeasured_ids]
                unmeasured_uncert = P_err[unmeasured_ids]
                sorted_unmeasured_ids=unmeasured_ids[np.argsort(unmeasured_P)]
                new_lig_ids=sorted_unmeasured_ids[:n_picks_per_step]
                
        else:
            raise(Exception(f"Unrecognized selection rule: {sele_rule}"))
        
        #5.8 finish up
        new_lig_ids=np.array(new_lig_ids, dtype=np.uint)
        step_lig_ids.append(new_lig_ids)
        known_lig_ids=np.concatenate((known_lig_ids, new_lig_ids))
        
        
        
        #5.9 record AL_metrics
        if(len(learner_metrics.top10_found)<=step): # only write learner metrics for this step if they aren't already there
            if(verbose):
                print(f"Computing metrics for step {step}")
        
            if(need_training):
                #load 2D_3D and calc metrics from it
                with open(sfiles[0], 'rb') as f:
                    settings_loaded, metrics = pickle.load(f)
                    bi=np.argmin(metrics.loss_XVal)
                    for m in learner_metrics.metrics_auto:
                        atrname=m+'_'+'XVal'
                        getattr(learner_metrics, atrname).append(getattr(metrics, atrname)[bi])
                    
                    if(select_on_summary_model):
                        P_all=metrics.summary_model_best_pred[0]
                    else:
                        P_all=metrics.best_pred[0]
                    P_all_errs=metrics.best_pred[1]
                    
                    def computeMeasures(dGs, Ps, P_errs, d):
                        high_binder_cutoff=-5*np.log(10)
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            se=np.square(dGs-Ps)
                            RMSE=np.sqrt(np.mean(se))
                            Cor=np.corrcoef(dGs, Ps)[0,1]
                            KT=sp.stats.kendalltau(dGs, Ps)[0]
                            FPR,TPR,_=get_FPR_TPR_AUC(dGs, Ps, high_binder_cutoff)
                            prec=get_precision(dGs, Ps, high_binder_cutoff)
                            AUC=get_fixed_ROC_AUC(dGs, Ps, P_errs, high_binder_cutoff)
                        
                        getattr(learner_metrics, f"RMSD_{d}").append(RMSE)
                        getattr(learner_metrics, f"Cor_{d}").append(Cor)
                        getattr(learner_metrics, f"KT_{d}").append(KT)
                        getattr(learner_metrics, f"TP_{d}").append(TPR)
                        getattr(learner_metrics, f"FP_{d}").append(FPR)
                        getattr(learner_metrics, f"prec_{d}").append(prec)
                        getattr(learner_metrics, f"AUC_{d}").append(AUC)
                    
                    last_selected=step_lig_ids[-1]
                    dGs=dG_all[last_selected]
                    Ps=P_all[last_selected]
                    P_errs=P_all_errs[last_selected]
                    computeMeasures(dGs, Ps, P_errs, "selected")
                                    
                    dGs=dG_all[unmeasured_ids]
                    Ps=P_all[unmeasured_ids]
                    P_errs=P_all_errs[unmeasured_ids]
                    computeMeasures(dGs, Ps, P_errs, "unmeasured")
                    
                    dGs=dG_all[top10_ids]
                    Ps=P_all[top10_ids]
                    P_errs=P_all_errs[top10_ids]
                    computeMeasures(dGs, Ps, P_errs, "top10")
                    
                    dGs=dG_all[top50_ids]
                    Ps=P_all[top50_ids]
                    P_errs=P_all_errs[top50_ids]
                    computeMeasures(dGs, Ps, P_errs, "top50")
                    
                    dGs=dG_all[top244_ids]
                    Ps=P_all[top244_ids]
                    P_errs=P_all_errs[top244_ids]
                    computeMeasures(dGs, Ps, P_errs, "top244")
                
                
            learner_metrics.top10_found.append(sum(el in known_lig_ids for el in top10_ids))
            learner_metrics.top50_found.append(sum(el in known_lig_ids for el in top50_ids))
            learner_metrics.top244_found.append(sum(el in known_lig_ids for el in top244_ids))
            write_AL_metrics_toHDD=True

            learner_metrics.known_lig_ids=known_lig_ids
            learner_metrics.step_lig_ids=step_lig_ids
            if(fraction_of_actives_to_use<1.0):
                learner_metrics.lig_ids_used_in_library=used_ids
            
        #5.10 save predictions
        if(save_Preds):
            #initialise prediction storage
            if not hasattr(learner_metrics, 'Predictions'):
                setattr(learner_metrics, 'Predictions', [])
                setattr(learner_metrics, 'Predictions_errs', [])
                
            #load 2D_3D and calc metrics from it
            with open(sfiles[0], 'rb') as f:
                settings_loaded, metrics = pickle.load(f)
                bi=np.argmin(metrics.loss_XVal)
                for m in learner_metrics.metrics_auto:
                    atrname=m+'_'+'XVal'
                    getattr(learner_metrics, atrname).append(getattr(metrics, atrname)[bi])
                
                if(select_on_summary_model):
                    P_all=metrics.summary_model_best_pred[0]
                else:
                    P_all=metrics.best_pred[0]
                P_all_errs=metrics.best_pred[1]
                
                learner_metrics.Predictions.append(P_all)
                learner_metrics.Predictions_errs.append(P_all_errs)
                print("setting Predictions for iter.", step)
        
            
        #5.11 save current progress
        if(write_AL_metrics_toHDD): #save only if there is something new in the metrics
            pickle.dump( (learner_settings, learner_metrics), open( settings_fn, "wb" ) )
            print("Saving settings at iter.", step)
        
        #if(step>2):
            #raise()
    
    return()


################################################################################

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
################################################################################
#start of execution
if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description='Does active learning on the experimental dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', dest='settings_fname', type=str, nargs='+',
                       help='settings pickle file')
    parser.add_argument('--redo', dest='redo', type=str2bool, nargs='?', const=True, default=False,
                        help="Recalculate even if output already exists.")
    parser.add_argument('--ligsf', dest='ligsf',
                default=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_with_2D_no_core_tSNE_and_step2_dG.pickle",
                help="Pickle file with full ligands and their simulated dG values.")
    parser.add_argument('--iter_all', dest='iter_all', action='store_True', default=False,
                        help="Should we iterate through the whole dataset?")
    
    args = parser.parse_args()
    
    ##################################################################################
    print("is CUDA available:", torch.cuda.is_available())

    def get_freer_gpu():
        cmd="nvidia-smi -q -d UTILIZATION | grep -A5 GPU | grep 'Gpu'"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        res = ps.communicate()[0].decode('utf-8').splitlines()
        gpu_load = [int(x.split()[-2]) for x in res]
        print("GPU loads are:",gpu_load, "%")
        return np.argmin(gpu_load)


    if torch.cuda.is_available() and not args.cpu_only:
        GPU_ID_to_use=None
        if('GPU_DEVICE_ORDINAL' in os.environ):
            GPU_ID_to_use = os.environ['GPU_DEVICE_ORDINAL'] # set by SLURM
        if(GPU_ID_to_use is None): #SGE does not set this
            GPU_ID_to_use = get_freer_gpu()
            print("Will use GPU id:", GPU_ID_to_use)
        else: #SLURM not only sets this, but also restricts what GPUs are even visible.
            #so we will always be using cuda:0 with SLURM, as that is always the one that SLURM assigns
            GPU_ID_to_use = 0
            print("SLURM detected, will use GPU id:", GPU_ID_to_use)
        dev = "cuda:"+str(GPU_ID_to_use)
        
    else:  
        dev = "cpu"
        print("Will use CPU only")
    device = torch.device(dev)
    print("Will train on device:", dev)

    n_cpus = int(os.environ['NSLOTS'])
    torch.set_num_threads(n_cpus)
    print("Will use this many CPU threads:", n_cpus)
    ##################################################################################



    for f in args.settings_fname:
        try:
            AL_Trainer(f, lig_db_fn=args.ligsf, redo=args.redo, iter_through_full_DSet=args.iter_all)
                              
            torch.cuda.empty_cache()
        except Exception as e:
            sys.stderr.write(repr(e)+'\n')
            
    exit(0);
