from tqdm import tqdm, trange
from copy import deepcopy

import matplotlib
if __name__== "__main__":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
import warnings
from datetime import datetime

#from IPython import display
#from IPython.display import clear_output
import copy
from copy import deepcopy
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#from captum.attr import IntegratedGradients
#from captum.attr import LayerConductance
#from captum.attr import NeuronConductance

import argparse
from sklearn.model_selection import KFold

try:
    import cPickle as pickle
except:
    import pickle
    
super_folder="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4"
pocket_fit_folder=f"{super_folder}/morphing_annealing_4d09_filtered_subset/"
#save_folder_base=f"{pocket_fit_folder}/prediction_step_2/meta_param_search"
script_folder=f"{pocket_fit_folder}/prediction_step_0/repr_scan_step_0_on_preliminary_calc_ddG"

#all_ligs_db_file=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_with_2D_no_core_tSNE_and_step2_dG.pickle"
#all_no_core_ligs_db_file=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_no_core_ligs_with_step2_dG.pickle"

Bohr2Ang=0.529177249
RT=0.001985875*300 #kcal/mol

import sys
#sys.path.insert(0, f"{script_folder}/..") 
#sys.path.insert(0, script_folder)
sys.path.append(f"{script_folder}/..")
sys.path.append(script_folder)

#import utils
from utils import *

#build NN class
import NNs
from NNs import *

#build data loader
import custom_dataset_modular_with_binning
from custom_dataset_modular_with_binning import dataBlocks, CustomMolModularDataset


    
    
class MultiTrainingRecord_owl:
    def __init__(self):
        self.epoch=[]
        
        self.metrics=["loss", "RMSD", "Cor", "TP", "FP", "AUC"]
        self.datasubsets=["Train", "Train_err", "XVal", "unknown", "Val", "Val_err"]
        #self.datasubsets=["Train", "Train_err", "XVal", "unknown", "Val"]
        
        self.sm_datasubsets=["sm_Train", "sm_Val", "sm_Train_err", "sm_Val_err"]
        #self.sm_datasubsets=["sm_Train", "sm_Val", "sm_Train_err"]
        
        for m in self.metrics:
            for d in self.datasubsets:
                setattr(self, m+'_'+d, [])
                
            for d in self.sm_datasubsets:
                setattr(self, m+'_'+d, [])
                
        self.kendalltau_XVal=[]
        self.final_pred=None
        self.best_pred=None
        
        self.summary_model_final_pred=None
        self.summary_model_best_pred=None


def train_model_with_settings_general_ensemble(settings_fname, redo=False, device=torch.device("cpu"), show_progress=False,
                              datafolder="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/prediction_step_0",
                              save_folder_base=f"{pocket_fit_folder}/prediction_step_2/meta_param_search",
                              all_ligs_db_file=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_with_2D_no_core_tSNE_and_step2_dG.pickle",
                              #all_no_core_ligs_db_file=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_no_core_ligs_with_step2_dG.pickle",
                              all_no_core_ligs_db_file=None,
                              validation_db_file=None,
                              no_core_validation_db_file=None,
                              train_summary_model=False,
                              max_cache_Mem_MB=512, predict_all_ligs=True,
                              training_ligs_idxs_in_all_ligs=None):
    print("\n\n\n")
    
    #load the settings
    with open(settings_fname, 'rb') as f:
        settings, metrics = pickle.load(f)
    
    if(metrics is not None and not redo):
        print(f"{settings_fname} is already trained. Will NOT retrain it.")
        return;
    
    representation_flags, dr_name, normalize_x, shuffle_seed, n_Epochs, hl_w, hl_depth, init_learning_rate, learning_rate_decay, weight_decay, impfilt, X_filter, weighted, shiftY, use_dropout  = settings
    
    #impfilt=None
    #X_filter=None
    
    print("Representation:", dr_name)
    print("NN width:", hl_w)
    print("NN depth:", hl_depth)
    print("Normalization of features:", normalize_x)
    print("Feature filtering:", impfilt)
    print("Loss weighted by dG frequency:", weighted)
    print("Prediction dG shifted and rescaled:", shiftY)
    print(datetime.now(), flush=True)
    
    if(not predict_all_ligs): # this is intended for only providing the training ligands to remote nodes running this script.
        # So normalization can't be done from only them.
        # Check for normalization cache already existing.
        #repr_hash=str(abs(hash(np.array(representation_flags, dtype=int).tobytes())))[:5]
        repr_hash=hashlib.md5(np.packbits(np.array(representation_flags, dtype=bool)).tobytes()).hexdigest()
        cachefolder=f"{datafolder}/combined_modular_repr_cache/{repr_hash}"
        filt_spec="_no_X_filter"
        fn_no_filt=f"{cachefolder}/normalization_factors_{filt_spec}.dat"
        if(not os.path.exists(fn_no_filt)):
            raise(Exception("Could not find normalization cache at:", fn_no_filt))
    
    #temp_dir = tempfile.TemporaryDirectory()
    #if()
    #print('Copied cached representation locally')
    
    # load set_0 known ligands and make dataloader
    #with open(step0_db_file, 'rb') as f:
    #    ligs = pickle.load(f)
        
    with open(all_ligs_db_file, 'rb') as f:
        all_ligs = pickle.load(f)
        
    #with open(all_no_core_ligs_db_file, 'rb') as f:
        #no_core_all_ligs = pickle.load(f)
        
    print("Read ligands", datetime.now(), flush=True)

    #train on all known ligs from dataset. Check if this includes validation ligands
    #training_ligs_idxs_in_all_ligs=[i for i in range(len(all_ligs)) if all_ligs[i].HasProp("dG")]    
    if(training_ligs_idxs_in_all_ligs is None):
        raise(Exception("indeces of the training ligands have not been provided"))
    ligs=[all_ligs[i] for i in training_ligs_idxs_in_all_ligs]
    #no_core_ligs=[no_core_all_ligs[i] for i in training_ligs_idxs_in_all_ligs]
    
    print("# of training ligands found:", len(ligs), datetime.now(), flush=True)
    #raise()
        
    lig_idxs_in_all=[]
    lig_idxs_in_known=[]
    known_lig_idxs=[l.GetProp('ID') for l in ligs]
    for i,lig in enumerate(all_ligs):
        if(lig.GetProp('ID') in known_lig_idxs):
            lig_idxs_in_all.append(i)
            lig_idxs_in_known.append(known_lig_idxs.index(lig.GetProp('ID')))
    
    #no_core_ligs=[None]*len(ligs)
    #for i in range(len(ligs)):
        #no_core_ligs[lig_idxs_in_known[i]]=no_core_all_ligs[lig_idxs_in_all[i]]
    
    high_binder_cutoff=-5*np.log(10)

    #XVal division function
    XVal_blocks=5
    def data_split(data):
        kf=KFold(n_splits=XVal_blocks, shuffle=True, random_state=12345)
        return(kf.split(data))

    #allYdata=-RT*(np.array([float(lig.GetProp('[V] hPDE2_pIC50')) for lig in ligs]))*np.log(10)
    allYdata=(np.array([float(lig.GetProp('dG')) for lig in ligs]))
    minY=np.min(allYdata)
    maxY=np.max(allYdata)

        #shuffle ligand indeces
    all_idxs_shuffled=np.arange(len(ligs))
    np.random.seed(shuffle_seed)
    np.random.shuffle(all_idxs_shuffled) # in place transform

    training_half=all_idxs_shuffled
    np.random.shuffle(training_half)
    n_starting_ligs=len(training_half)
    print("# starting ligands:",n_starting_ligs, datetime.now(), flush=True)
    known_idxs=training_half[:n_starting_ligs]
    unknown_idxs=training_half[n_starting_ligs:]


    # detailed params

    batchsize=500
    eval_batchsize=200
    activation="relu"

    eval_every=min(10, n_Epochs)
    #plot_every=min(100, n_Epochs)
    plot_every=min(1000, n_Epochs)
    backup_models_every=1000

    if(X_filter is not None):
        if(not os.path.exists(X_filter)):
            raise(Exception(f"No such file: {X_filter}"))


    if(weighted):
        kde = sp.stats.gaussian_kde(allYdata)
        temp_Y=np.linspace(np.min(allYdata), np.max(allYdata), 20, endpoint=True)
        temp_kde=kde(temp_Y)
        C=1/np.mean(1/temp_kde)
        def get_weights(y):
            return(C/kde(y))
        weight_func=get_weights
        
        ###################debug###################
        #weights=weight_func(temp_Y)
        #print(temp_Y)
        #print(weights)
        #plt.plot(temp_Y, weights)
        #plt.show()
        #raise()
        ###################debug###################
    else:
        weight_func=None
        
    if(use_dropout):
        p_dropout=np.repeat(0.5, hl_depth+1)
    else:
        p_dropout=np.repeat(0.0, hl_depth+1)
        
    noise=0.0
    
    print("Finished setting up weights", datetime.now(), flush=True)
        

    train_clr="blue"
    Xval_clr="purple"
    unkn_clr="green"
    summary_train_clr="darkred"
    summary_unkn_clr="peru"


    #generator for all data
    full_dataset = CustomMolModularDataset(ligs=all_ligs,
                                           #no_core_ligs=no_core_all_ligs,
                                           no_core_ligs=None,
                                           use_combined_cache=False, use_hdf5_cache=False,
                                           representation_flags=representation_flags, normalize_x=normalize_x,
                                           X_filter=X_filter, datafolder=datafolder,
                                           internal_cache_maxMem_MB= max_cache_Mem_MB if predict_all_ligs else 0)
    if(normalize_x):
        full_dataset.find_normalization_factors()
        print("Found normalization factors across all ligands", datetime.now(), flush=True)
    if(predict_all_ligs):
        all_generator = torch.utils.data.DataLoader(full_dataset, shuffle=False, batch_size=eval_batchsize)
        
        
    # Set up validation dataset & generator
    if(validation_db_file is not None):
        #if(no_core_validation_db_file is None):
            #raise(Exception("Please provide both validation core and non-core pickles."))
        
        with open(validation_db_file, 'rb') as f:
            val_ligs = pickle.load(f)
        
        #with open(no_core_validation_db_file, 'rb') as f:
            #no_core_val_ligs = pickle.load(f)
            
        print(f"There are {len(val_ligs)} validation ligands")
        
        val_DS = CustomMolModularDataset(ligs=val_ligs,
                                #no_core_ligs=no_core_val_ligs,
                                no_core_ligs=None,
                                representation_flags=representation_flags, normalize_x=False,
                                use_combined_cache=True,
                                X_filter=X_filter, datafolder=datafolder)
        val_DS.copy_normalization_factors(full_dataset)
        val_DS.normalize_x=True
        val_DS.build_internal_filtered_cache()
        
        val_generator = torch.utils.data.DataLoader(val_DS, shuffle=False, batch_size=eval_batchsize)   
        
        n_val_ligs = len(val_ligs)
        
        del val_ligs, val_DS#, no_core_val_ligs
        _=gc.collect()
        
        print("Set up validation dataset", datetime.now(), flush=True)
        
            
    multi_model_record=None
    global_models=None
    global_summary_model=None
    
    
    dr_detailed_name=dr_name if not "ESP" in dr_name else dr_name+"_p{}_a{}".format(full_dataset.grid_padding, full_dataset.grid_spacing)
    
    
    # WARNING: below hashes are not reproducible as python version and pickle protocol id change
    # see https://death.andgravity.com/stable-hashing
    sha = hashlib.sha256()
    sha.update(pickle.dumps(settings))
    settings_hash=sha.hexdigest()[:10]
        
    save_folder=save_folder_base+f"/models/{dr_name}/"
    save_folder+=f"known_{len(known_idxs)}_width_{hl_w}_depth_{hl_depth}"+\
                    f"_trd_{learning_rate_decay}_wd_{weight_decay}"+("" if impfilt is None else f"_impfilt_{impfilt}")+\
                    ('' if not normalize_x else '_norm_X')+("" if not weighted else "_weightedbyY")+("" if not use_dropout else f"_dropout{p_dropout[0]}")+\
                    ('' if noise<=0.0 else f"_noise_{noise}")+("_shiftY" if shiftY else "")+f"_{n_Epochs}ep"
    os.makedirs(save_folder, exist_ok=True)
    save_plot=save_folder+f"/summary_{settings_hash}_ep_{n_Epochs}.png"
    
    print("Will save models in:\n\t", save_folder)
       
   
    #################
    # do the training
    #################
    print("Start setting up models", datetime.now(), flush=True)
    
    if(predict_all_ligs):
        n_features = full_dataset[0][0].shape[-1]
    else:
        #no point normalising or building internal cache for this one, we just want to know # of features it spits out
        DS = CustomMolModularDataset([ligs[0]],
                                     #no_core_ligs=[no_core_ligs[0]],
                                     no_core_ligs=None,
                                     representation_flags=representation_flags, normalize_x=False,
                                     X_filter=X_filter, datafolder=datafolder,
                                     internal_cache_maxMem_MB=0)
        n_features = DS[0][0].shape[-1]
        del DS
        _=gc.collect()
    known_Y=allYdata[known_idxs]
    print("There are", n_features, "features", datetime.now(), flush=True)

    models=[]
    model_Train_idxs=[]
    model_XVal_idxs=[]
    split_gen=data_split(known_idxs)
    for drop in range(XVal_blocks):
        
        #create model
        model = Net(inp_width=n_features, hl_w=hl_w, nhl=hl_depth, activation=activation,
                    learning_rate=init_learning_rate, lr_decay=learning_rate_decay,
                    high_binder_cutoff=high_binder_cutoff, weight_decay=weight_decay,
                    weights_distrib_func=weight_func, drop_p=p_dropout, noise=noise, shiftY=shiftY).to(device)
        
        #create training, cross validation, and test data
        Train_indeces, XVal_indeces = next(split_gen)
        
        #save as indexes within known_idxs array
        model_Train_idxs.append(Train_indeces)
        model_XVal_idxs.append(XVal_indeces)
        
        Train_indeces = known_idxs[Train_indeces] #convert indexing into the ligs array
        XVal_indeces  = known_idxs[XVal_indeces]  #convert indexing into the ligs array

        Train_set = CustomMolModularDataset([ligs[idx] for idx in Train_indeces],
                                            #no_core_ligs=[no_core_ligs[idx] for idx in Train_indeces],
                                            no_core_ligs=None,
                                            representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter, datafolder=datafolder,
                                            internal_cache_maxMem_MB=max_cache_Mem_MB)
        XVal_set  = CustomMolModularDataset([ligs[idx] for idx in XVal_indeces ],
                                            #no_core_ligs=[no_core_ligs[idx] for idx in Train_indeces],
                                            no_core_ligs=None,
                                            representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter, datafolder=datafolder,
                                            internal_cache_maxMem_MB=max_cache_Mem_MB)
        Train_set.copy_normalization_factors(full_dataset)
        XVal_set.copy_normalization_factors(full_dataset)
        
        #build_internal_filtered_cache
        Train_set.build_internal_filtered_cache()
        XVal_set.build_internal_filtered_cache()
        
        #create training, cross validation, and test datagens
        model.training_generator            = torch.utils.data.DataLoader(Train_set, shuffle=True,  batch_size=batchsize)
        model.training_generator_no_shuffle = torch.utils.data.DataLoader(Train_set, shuffle=False, batch_size=eval_batchsize)
        model.crossvalidation_generator     = torch.utils.data.DataLoader(XVal_set,  shuffle=False, batch_size=eval_batchsize)
        
        #add tracking of validation ligands
        if(validation_db_file is not None):
            model.test_generator            = val_generator
        
        del Train_set, XVal_set
        
        models.append(model)
        print("Finished setup for model", drop, datetime.now())
        
    summary_models=[]
    Train_set = CustomMolModularDataset([ligs[idx] for idx in known_idxs],
                                        #no_core_ligs=[no_core_ligs[idx] for idx in known_idxs],
                                        no_core_ligs=None,
                                        representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter, datafolder=datafolder,
                                        internal_cache_maxMem_MB=max_cache_Mem_MB)
    Train_set.copy_normalization_factors(full_dataset)
    Train_set.build_internal_filtered_cache()
    sm_gen_shuffle=torch.utils.data.DataLoader(Train_set, shuffle=True,  batch_size=batchsize)
    sm_gen_no_shuffle=torch.utils.data.DataLoader(Train_set, shuffle=False,  batch_size=batchsize)
    
    if(train_summary_model):
        for drop in range(XVal_blocks):
            sm = Net(inp_width=n_features, hl_w=hl_w, nhl=hl_depth, activation=activation,
                    learning_rate=init_learning_rate, lr_decay=learning_rate_decay,
                    high_binder_cutoff=high_binder_cutoff, weight_decay=weight_decay,
                    weights_distrib_func=weight_func, drop_p=p_dropout, noise=noise, shiftY=shiftY).to(device)
        
            sm.training_generator            = sm_gen_shuffle
            sm.training_generator_no_shuffle = sm_gen_no_shuffle
            
            #add tracking of validation ligands
            if(validation_db_file is not None):
                sm.test_generator            = val_generator
                
            summary_models.append(sm)
        
        print("Finished setup for summary model", drop, datetime.now())
    del Train_set

    multi_model_record = MultiTrainingRecord_owl()
    if (validation_db_file is None):
        print("There is no validation dataset provided. Disabling validation metrics.")
        multi_model_record.datasubsets=["Train", "Train_err", "XVal"]
        multi_model_record.sm_datasubsets=["sm_Train", "sm_Train_err"]

    print("Training for", n_Epochs, "epochs", datetime.now(), flush=True)
    myrange=range(0, n_Epochs)
    #if(show_progress):
        #myrange=trange(0, n_Epochs, desc="Epoch")
    #for ep in trange(0, n_Epochs, desc="Epoch"):
    #for ep in range(0, n_Epochs):
    for ep in myrange:
        #train K_fold models
        for drop in range(XVal_blocks):
            models[drop].train_epoch(eval_every=eval_every)
            
            if(train_summary_model):
                summary_models[drop].train_epoch(eval_every=eval_every)

        if ep%eval_every == eval_every-1 or ep == 0:            
            all_Y_known=torch.zeros((len(known_idxs), 1), dtype=torch.float, device=device)
            multi_P_XVal=torch.zeros((len(known_idxs), 1), dtype=torch.float, device=device)
#             multi_P_eval=torch.zeros((len(evaluation_half), 1, XVal_blocks), dtype=torch.float, device=device)
            for drop in range(XVal_blocks):
                test_ndxs=model_XVal_idxs[drop]
                P_XVal, Y_XVal = models[drop].get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                multi_P_XVal[test_ndxs,:]= P_XVal
                all_Y_known[test_ndxs,:] = Y_XVal
                        
            np_all_Y_known=all_Y_known.detach().cpu().numpy()[:,0]
            np_multi_P_XVal=multi_P_XVal.detach().cpu().numpy()[:,0]
                
            XVal_loss=models[0].loss_fn(all_Y_known, multi_P_XVal).detach().cpu().tolist()
            XVal_RMSE=torch.sqrt(models[0].loss_mse(all_Y_known, multi_P_XVal)).detach().cpu().numpy()
            XVal_Cor=np.corrcoef(np_all_Y_known.flatten(), np_multi_P_XVal.flatten())[0,1]
            XVal_KT=sp.stats.kendalltau(np_all_Y_known.flatten(), np_multi_P_XVal.flatten())[0]
            
            
            #true and false positive rates + AUC
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #XVal_FP, XVal_TP, XVal_AUC = get_FPR_TPR_AUC(np_all_Y_known, np_multi_P_XVal, high_binder_cutoff)
                XVal_FP, XVal_TP = get_FPR_and_TPR(np_all_Y_known, np_multi_P_XVal, high_binder_cutoff)
                XVal_AUC = get_fixed_ROC_AUC(np_all_Y_known, np_multi_P_XVal, P_err=None, cut=high_binder_cutoff)
            
            #save state at best XVal loss
            if(ep>0 and XVal_loss < np.min(multi_model_record.loss_XVal)):
                for m in models:
                    m.cache_state()

            #95 % confidence interval assuming gaussian error model
            #confinterval= lambda a: st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))[0]-np.mean(a)

            multi_model_record.epoch.append(ep)
            multi_model_record.loss_XVal.append(XVal_loss)
            multi_model_record.RMSD_XVal.append(XVal_RMSE)
            multi_model_record.Cor_XVal.append(XVal_Cor)
            multi_model_record.kendalltau_XVal.append(XVal_KT)
            
            multi_model_record.TP_XVal.append(XVal_TP)
            multi_model_record.FP_XVal.append(XVal_FP)
            multi_model_record.AUC_XVal.append(XVal_AUC)
                        
            #all the other metrics taken from individual models
            func_mean = lambda n : np.mean([getattr(m.record, n)[-1] for m in models])
            func_err = lambda n : confinterval([getattr(m.record, n)[-1] for m in models])
            for d in multi_model_record.datasubsets:
                srcd=d
                f=func_mean
                if(d=="XVal" or "summary_model" in d or d=="unknown"):
                    continue; #these are set manually
                if((d=="Val" or d=="Val_err") and validation_db_file is None):
                    continue; # can't record validation metrics if there is no validation dataset
                if ("_err" in d):
                    srcd=d[:-4]
                    f=func_err
                
                for metric in multi_model_record.metrics:
                    atrname=metric+'_'+d
                    srcname=metric+'_'+srcd
                    try:
                        getattr(multi_model_record, atrname).append(f(srcname))
                    except Exception as e:
                        print(f"d={d}\t atrname={atrname}\t srcname={srcname}", flush=True)
                        raise(e);
                    
            if(train_summary_model):
                sm_func_mean = lambda n : np.mean([getattr(sm.record, n)[-1] for sm in summary_models])
                sm_func_err = lambda n : confinterval([getattr(sm.record, n)[-1] for sm in summary_models])
                for d in multi_model_record.sm_datasubsets:
                    for metric in multi_model_record.metrics:
                        srcd=d[3:]
                        f=sm_func_mean
                        if ("_err" in d):
                            srcd=d[3:-4]
                            f=sm_func_err
                        
                        atrname=metric+'_'+d
                        srcname=metric+'_'+srcd
                        getattr(multi_model_record, atrname).append(f(srcname))
            
            
            #save state at best Val loss for summary_model
            if(train_summary_model):
                if(ep>0 and multi_model_record.loss_sm_Val[-1] < np.min(multi_model_record.loss_sm_Val[:-1])):
                    for sm in summary_models:
                        sm.cache_state()
                    
                
                    
            if(show_progress):
                if(validation_db_file is None):
                    print(f"\tEpoch: {ep}\t train_loss={multi_model_record.loss_Train[-1]}\t XVal_RMSE={XVal_RMSE}", end="")
                else:
                    print(f"\tEpoch: {ep}\t train_loss={multi_model_record.loss_Train[-1]:.4f}+-{multi_model_record.loss_Train_err[-1]:.4f}\t XVal_RMSE={XVal_RMSE:.4f}\t Val_RMSE={multi_model_record.RMSD_Val[-1]:.4f}+-{multi_model_record.RMSD_Val_err[-1]:.4f}", end="")
                    
                if(train_summary_model):
                    if(validation_db_file is None):
                        print(f"\t sm_train_loss={multi_model_record.loss_sm_Train[-1]:.4}", end="")
                    else:
                        print(f"\t sm_train_loss={multi_model_record.loss_sm_Train[-1]:.4}\t sm_Val_RMSE={multi_model_record.RMSD_sm_Val[-1]:.4f}", end="")
                        
                print("", flush=True)
                
                ###################debug###################
                ## individual model losses are:
                #print("\t\tL1 losses for models are:\t", end="")
                #for drop in range(XVal_blocks): #go through each block
                #    P, Y = models[drop].get_predictions_from_batches(models[drop].training_generator, with_np=False)
                #    loss=nn.L1Loss()(P,Y).detach().tolist()
                #    print(f"m{drop}: {loss:.4f}\t", end="")
                #    
                #P, Y = summary_model.get_predictions_from_batches(summary_model.training_generator, with_np=False)
                #loss=nn.L1Loss()(P,Y).detach().tolist()
                #print(f"summary_model: {loss:.4f}\t")
                #
                #print("\t\tlosses in records are:\t", end="")
                #for drop in range(XVal_blocks): #go through each block
                #    print(f"m{drop}: {models[drop].record.loss_Train[-1]:.4f}\t", end="")                    
                #print(f"summary_model: {summary_model.record.loss_Train[-1]:.4f}\t")
                ###################debug###################
                
        # check for nans and restart training if detected
        if(not np.isfinite(multi_model_record.loss_Train[-1]) or not np.isfinite(XVal_RMSE)):
            raise(RuntimeError("Nan in metrics"))                    
                        
        #plot all models at once
        if ep%plot_every == plot_every-1:

            fig=plt.figure(figsize=(20,8))
            sp_grid = gridspec.GridSpec(2, 5)

            ax = plt.subplot(sp_grid[0,0])
            plt.ylim(0, 2.5)
            rec = multi_model_record
            bi=np.argmin(rec.loss_XVal)
            if(validation_db_file is not None):
                sm_bi=np.argmin(rec.loss_sm_Val)

            ax.errorbar(rec.epoch[::10], rec.loss_Train[::10], yerr=rec.loss_Train_err[::10], color=train_clr, alpha=0.3, linestyle="none")
            ax.plot(rec.epoch, rec.loss_Train, alpha=0.3, label="training", color=train_clr)
            ax.plot(rec.epoch, rec.loss_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.scatter(rec.epoch[bi], rec.loss_Train[bi], color=train_clr, marker="*")
            ax.scatter(rec.epoch[bi], rec.loss_XVal[bi], color=Xval_clr, marker="*")
            eprange=[0,ep]
            MAE_best=[rec.loss_XVal[bi],rec.loss_XVal[bi]]
            MAE_last=[rec.loss_XVal[-1],rec.loss_XVal[-1]]
            ax.plot(eprange, MAE_best, '--k', zorder=0, alpha=0.3, label=None)
            ax.plot(eprange, MAE_last, '--k', zorder=0, alpha=0.3, label=None)
            if(validation_db_file is not None):
                ax.errorbar(rec.epoch[::10], rec.loss_Val[::10], yerr=rec.loss_Val_err[::10], color=unkn_clr, alpha=0.3, linestyle="none")
                ax.plot(rec.epoch, rec.loss_Val, alpha=0.3, label="expt. validation", color=unkn_clr)
                ax.scatter(rec.epoch[bi], rec.loss_Val[bi], color=unkn_clr, marker="*")
            if(train_summary_model):
                ax.errorbar(rec.epoch[::10], rec.loss_sm_Train[::10], yerr=rec.loss_sm_Train_err[::10], color=summary_train_clr, alpha=0.3, linestyle="none")
                ax.plot(rec.epoch, rec.loss_sm_Train, alpha=0.3, label="s.m. training", color=summary_train_clr)
                ax.scatter(rec.epoch[sm_bi], rec.loss_sm_Train[sm_bi], color=summary_train_clr, marker="X")
                if(validation_db_file is not None):
                    ax.errorbar(rec.epoch[::10], rec.loss_sm_Val[::10], yerr=rec.loss_sm_Val_err[::10], color=summary_unkn_clr, alpha=0.3, linestyle="none")
                    ax.plot(rec.epoch, rec.loss_sm_Val, alpha=0.3, label="s.m. expt. validation", color=summary_unkn_clr)
                    ax.scatter(rec.epoch[sm_bi], rec.loss_sm_Val[sm_bi], color=summary_unkn_clr, marker="X")
                

            ax.set(xlabel='Epoch', ylabel='Loss (MAE + Penalties)')

            ax = plt.subplot(sp_grid[0,1])
            plt.ylim(0, 2.5)
            ax.errorbar(rec.epoch[::10], rec.RMSD_Train[::10], yerr=rec.RMSD_Train_err[::10], color=train_clr, alpha=0.3, linestyle="none")
            ax.plot(rec.epoch, rec.RMSD_Train, alpha=0.3, label="training", color=train_clr)
            ax.plot(rec.epoch, rec.RMSD_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.scatter(rec.epoch[bi], rec.RMSD_Train[bi], color=train_clr, marker="*")
            ax.scatter(rec.epoch[bi], rec.RMSD_XVal[bi], color=Xval_clr, marker="*")
            ax.set(xlabel='Epoch', ylabel='RMSD')
            if(validation_db_file is not None):
                ax.errorbar(rec.epoch[::10], rec.RMSD_Val[::10], yerr=rec.RMSD_Val_err[::10], color=unkn_clr, alpha=0.3, linestyle="none")
                ax.plot(rec.epoch, rec.RMSD_Val, alpha=0.3, label="expt. validation", color=unkn_clr)
                ax.scatter(rec.epoch[bi], rec.RMSD_Val[bi], color=unkn_clr, marker="*")
            if(train_summary_model):
                ax.errorbar(rec.epoch[::10], rec.RMSD_sm_Train[::10], yerr=rec.RMSD_sm_Train_err[::10], color=summary_train_clr, alpha=0.3, linestyle="none")
                ax.plot(rec.epoch, rec.RMSD_sm_Train, alpha=0.3, label="s.m. training", color=summary_train_clr)
                ax.scatter(rec.epoch[sm_bi], rec.RMSD_sm_Train[sm_bi], color=summary_train_clr, marker="X")
                if(validation_db_file is not None):
                    ax.errorbar(rec.epoch[::10], rec.RMSD_sm_Val[::10], yerr=rec.RMSD_sm_Val_err[::10], color=summary_unkn_clr, alpha=0.3, linestyle="none")
                    ax.plot(rec.epoch, rec.RMSD_sm_Val, alpha=0.3, label="s.m. expt. validation", color=summary_unkn_clr)
                    ax.scatter(rec.epoch[sm_bi], rec.RMSD_sm_Val[sm_bi], color=summary_unkn_clr, marker="X")

            ax = plt.subplot(sp_grid[0,2])
            plt.ylim(0, 1)
            ax.errorbar(rec.epoch[::10], rec.Cor_Train[::10], yerr=rec.Cor_Train_err[::10], color=train_clr, alpha=0.3, linestyle="none")
            ax.plot(rec.epoch, rec.Cor_Train, alpha=0.3, label="training", color=train_clr)
            ax.plot(rec.epoch, rec.Cor_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.scatter(rec.epoch[bi], rec.Cor_Train[bi], color=train_clr, marker="*")
            ax.scatter(rec.epoch[bi], rec.Cor_XVal[bi], color=Xval_clr, marker="*")
            ax.set(xlabel='Epoch', ylabel='Corelation')
            if(validation_db_file is not None):
                ax.errorbar(rec.epoch[::10], rec.Cor_Val[::10], yerr=rec.Cor_Val_err[::10], color=unkn_clr, alpha=0.3, linestyle="none")
                ax.plot(rec.epoch, rec.Cor_Val, alpha=0.3, label="expt. validation", color=unkn_clr)
                ax.scatter(rec.epoch[bi], rec.Cor_Val[bi], color=unkn_clr, marker="*")
            if(train_summary_model):
                ax.errorbar(rec.epoch[::10], rec.Cor_sm_Train[::10], yerr=rec.Cor_sm_Train_err[::10], color=summary_train_clr, alpha=0.3, linestyle="none")
                ax.plot(rec.epoch, rec.Cor_sm_Train, alpha=0.3, label="s.m. training", color=summary_train_clr)
                ax.scatter(rec.epoch[sm_bi], rec.Cor_sm_Train[sm_bi], color=summary_train_clr, marker="X")
                if(validation_db_file is not None):
                    ax.errorbar(rec.epoch[::10], rec.Cor_sm_Val[::10], yerr=rec.Cor_sm_Val_err[::10], color=summary_unkn_clr, alpha=0.3, linestyle="none")
                    ax.plot(rec.epoch, rec.Cor_sm_Val, alpha=0.3, label="s.m. expt. validation", color=summary_unkn_clr)
                    ax.scatter(rec.epoch[sm_bi], rec.Cor_sm_Val[sm_bi], color=summary_unkn_clr, marker="X")

            plt.legend()
            
            ################
            ax = plt.subplot(sp_grid[1,0])
            plt.ylim(0, 1)
            ax.plot(rec.epoch, rec.TP_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.errorbar(rec.epoch[::10], rec.TP_Train[::10], yerr=rec.TP_Train_err[::10], label="training", color=train_clr, alpha=0.3, linestyle="none")
            ax.set(xlabel='Epoch', ylabel='True Positive Rate')
            if(validation_db_file is not None):
                ax.errorbar(rec.epoch[::10], rec.TP_Val[::10], yerr=rec.TP_Val_err[::10], color=unkn_clr, alpha=0.3, label="expt. validation")
            if(train_summary_model):
                ax.errorbar(rec.epoch[::10], rec.TP_sm_Train[::10], yerr=rec.TP_sm_Train_err[::10], color=summary_train_clr, alpha=0.3, label="s.m. training")
                #ax.plot(rec.epoch, rec.TP_sm_Train, alpha=0.3, label="s.m. training", color=summary_train_clr)
                if(validation_db_file is not None):
                    ax.errorbar(rec.epoch[::10], rec.TP_sm_Val[::10], yerr=rec.TP_sm_Val_err[::10], color=summary_unkn_clr, alpha=0.3, label="s.m.  expt. validation")
                    #ax.plot(rec.epoch, rec.TP_sm_Val, alpha=0.3, label="s.m. expt. validation", color=summary_unkn_clr)
            
            ax = plt.subplot(sp_grid[1,1])
            plt.ylim(0, 1)
            ax.plot(rec.epoch, rec.FP_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.errorbar(rec.epoch[::10], rec.FP_Train[::10], yerr=rec.FP_Train_err[::10], label="training", color=train_clr, alpha=0.3, linestyle="none")
            ax.set(xlabel='Epoch', ylabel='False Positive Rate')
            if(validation_db_file is not None):
                ax.errorbar(rec.epoch[::10], rec.FP_Val[::10], yerr=rec.FP_Val_err[::10], color=unkn_clr, alpha=0.3, label="expt. validation")
            if(train_summary_model):
                ax.errorbar(rec.epoch[::10], rec.FP_sm_Train[::10], yerr=rec.FP_sm_Train_err[::10], color=summary_train_clr, alpha=0.3, label="s.m. training")
                #ax.plot(rec.epoch, rec.FP_sm_Train, alpha=0.3, label="s.m. training", color=summary_train_clr)
                if(validation_db_file is not None):
                    ax.errorbar(rec.epoch[::10], rec.FP_sm_Val[::10], yerr=rec.FP_sm_Val_err[::10], color=summary_unkn_clr, alpha=0.3, label="s.m. expt. validation")
                    #ax.plot(rec.epoch, rec.FP_sm_Val, alpha=0.3, label="s.m. expt. validation", color=summary_unkn_clr)

            
            ax = plt.subplot(sp_grid[1,2])
            plt.ylim(0, 1)
            ax.plot(rec.epoch, rec.AUC_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.errorbar(rec.epoch[::10], rec.AUC_Train[::10], yerr=rec.AUC_Train_err[::10], label="training", color=train_clr, alpha=0.3, linestyle="none")
            ax.set(xlabel='Epoch', ylabel='Area under ROC curve')
            if(validation_db_file is not None):
                ax.errorbar(rec.epoch[::10], rec.AUC_Val[::10], yerr=rec.AUC_Val_err[::10], color=unkn_clr, alpha=0.3, label="expt. validation")
            if(train_summary_model):
                ax.errorbar(rec.epoch[::10], rec.AUC_sm_Train[::10], yerr=rec.AUC_sm_Train_err[::10], color=summary_train_clr, alpha=0.3, label="s.m. training")
                #ax.plot(rec.epoch, rec.AUC_sm_Train, alpha=0.3, label="s.m. training", color=summary_train_clr)
                if(validation_db_file is not None):
                    ax.errorbar(rec.epoch[::10], rec.AUC_sm_Val[::10], yerr=rec.AUC_sm_Val_err[::10], color=summary_unkn_clr, alpha=0.3, label="s.m. expt. validation")
                    #ax.plot(rec.epoch, rec.AUC_sm_Val, alpha=0.3, label="s.m. expt. validation", color=summary_unkn_clr)

            
            

            #prepare data across multiple models            
            multi_P_XVal=torch.zeros(all_Y_known.shape, dtype=torch.float)
            best_P_XVal=torch.zeros(all_Y_known.shape, dtype=torch.float)
            multi_P_Train = np.zeros((all_Y_known.shape[0], XVal_blocks-1)) #indexed by [data, repeat]
            best_P_Train  = np.zeros((all_Y_known.shape[0], XVal_blocks-1)) #indexed by [data, repeat]

            for drop in range(XVal_blocks): #go through each block
                test_ndxs=model_XVal_idxs[drop]

                P_XVal, Y_XVal = models[drop].get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                multi_P_XVal[test_ndxs,:]=P_XVal.detach().cpu()
            
                models[drop].restore_state() #best epoch so far
                P_XVal, Y_XVal  = models[drop].get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                models[drop].restore_state() #reset to current epoch
                best_P_XVal[test_ndxs,:]=P_XVal.detach().cpu()



                #for training predictions,
                #go through each model
                for k, m in enumerate(models):               
                    j=k # j<drop
                    if(k==drop):
                        continue;
                    elif(k>drop):
                        j=k-1

                    P_train, Y_train = m.get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                    multi_P_Train[test_ndxs, j] = P_train.detach().cpu().numpy()[:,0]

                    m.restore_state() #best epoch so far
                    P_train, Y_train = m.get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                    m.restore_state() #reset to current epoch
                    best_P_Train[test_ndxs, j] = P_train.detach().cpu().numpy()[:,0]
          
            
            multi_P_XVal = multi_P_XVal.detach().cpu().numpy()[:,0] #convert to numpy, ordered like known_Y
            multi_P_Train_mean= np.mean(multi_P_Train, axis=-1) # ordered like known_Y
            multi_P_Train_err = [confinterval(multi_P_Train[g,:]) for g in range(multi_P_Train.shape[0])]

            best_P_XVal = best_P_XVal.detach().cpu().numpy()[:,0] #convert to numpy, ordered like known_Y
            best_P_Train_mean = np.mean(best_P_Train, axis=-1) # ordered like known_Y
            best_P_Train_err = [confinterval(best_P_Train[g,:]) for g in range(best_P_Train.shape[0])]
            
            
            if(validation_db_file is not None):
                multi_P_Val = np.zeros((n_val_ligs, XVal_blocks)) #indexed by [data, repeat]
                best_P_Val  = np.zeros((n_val_ligs, XVal_blocks)) #indexed by [data, repeat]
                
                for drop in range(XVal_blocks): #go through each block

                    P, Y_Val = models[drop].get_predictions_from_batches(models[drop].test_generator, with_np=False)
                    multi_P_Val[:,drop]=P.detach().cpu().numpy()[:,0]
                
                    models[drop].restore_state() #best epoch so far
                    P, Y_Val = models[drop].get_predictions_from_batches(models[drop].test_generator, with_np=False)
                    models[drop].restore_state() #reset to current epoch
                    best_P_Val[:,drop]=P.detach().cpu().numpy()[:,0]
                    
                multi_P_Val_mean=np.mean(multi_P_Val, axis=-1)
                multi_P_Val_err = np.array([confinterval(multi_P_Val[g,:]) for g in range(best_P_Val.shape[0])])
                best_P_Val_mean=np.mean(best_P_Val, axis=-1)
                best_P_Val_err = np.array([confinterval(best_P_Val[g,:]) for g in range(best_P_Val.shape[0])])
                Y_Val = Y_Val.detach().cpu().numpy()
                
                del multi_P_Val, best_P_Val
                    

            #plot data
            ax = plt.subplot(sp_grid[0,3])
            #lims=(-15,-4)
            #lims=(-18,-4)
            lims=(-25,-4)
            plt.ylim(lims[0], lims[1])
            plt.xlim(lims[0], lims[1])
            ax.errorbar(known_Y, multi_P_Train_mean, yerr=multi_P_Train_err, color=train_clr, marker=".", ls='', alpha=0.2, zorder=1)
            ax.scatter(known_Y, multi_P_XVal, color=Xval_clr, marker=".", alpha=0.4, zorder=2)
            if(validation_db_file is not None):
                ax.errorbar(Y_Val, multi_P_Val_mean, yerr=multi_P_Val_err, color=unkn_clr, marker=".", ls='', alpha=0.2, zorder=3)            
            diag = np.linspace(lims[0],lims[1], 10)
            plt.plot(diag, diag, '--k', zorder=0)
            plt.vlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            plt.hlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            ax.set(xlabel='dG Sim. or Expt. (kcal/mol)', ylabel='dG Predicted (kcal/mol)', title="Current Epoch")
            
            

            #plot best data
            ax = plt.subplot(sp_grid[0,4])
            plt.ylim(lims[0], lims[1])
            plt.xlim(lims[0], lims[1])
            ax.errorbar(known_Y, best_P_Train_mean, yerr=best_P_Train_err, color=train_clr, marker=".", ls='', alpha=0.2, zorder=1)
            ax.scatter(known_Y, best_P_XVal, color=Xval_clr, marker=".", alpha=0.4, zorder=2)
            if(validation_db_file is not None):
                ax.errorbar(Y_Val, best_P_Val_mean, yerr=best_P_Val_err, color=unkn_clr, marker=".", ls='', alpha=0.2, zorder=3)
            diag = np.linspace(lims[0],lims[1], 10)
            plt.plot(diag, diag, '--k', zorder=0)
            plt.vlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            plt.hlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            ax.set(xlabel='dG Sim. or Expt. (kcal/mol)', ylabel='dG Predicted (kcal/mol)', title="Best XVal. Epoch (star)")
            
            
            # plot currect and best summary_model data too
            if(train_summary_model):
                #gather summary model data
                sm_multi_P_Val = np.zeros((n_val_ligs, XVal_blocks)) #indexed by [data, repeat]
                sm_best_P_Val  = np.zeros((n_val_ligs, XVal_blocks)) #indexed by [data, repeat]
                sm_multi_P_Train = np.zeros((all_Y_known.shape[0], XVal_blocks)) #indexed by [data, repeat]
                sm_best_P_Train  = np.zeros((all_Y_known.shape[0], XVal_blocks)) #indexed by [data, repeat]
                
                for drop in range(XVal_blocks): #go through each block
                    summary_model=summary_models[drop]
                    
                    P, Y_Train = summary_model.get_predictions_from_batches(summary_model.training_generator_no_shuffle, with_np=False)
                    sm_multi_P_Train[:,drop]=P.detach().cpu().numpy()[:,0]
                    if(validation_db_file is not None): # validation
                        P, Y_Val = summary_model.get_predictions_from_batches(summary_model.test_generator, with_np=False)
                        sm_multi_P_Val[:,drop]=P.detach().cpu().numpy()[:,0]
                        
                    
                    summary_model.restore_state() #best epoch so far
                    P, Y_Train = summary_model.get_predictions_from_batches(summary_model.training_generator_no_shuffle, with_np=False)
                    sm_best_P_Train[:,drop]=P.detach().cpu().numpy()[:,0]
                    if(validation_db_file is not None): # validation
                        P, Y_Val = summary_model.get_predictions_from_batches(summary_model.test_generator, with_np=False)
                        sm_best_P_Val[:,drop]=P.detach().cpu().numpy()[:,0]
                    summary_model.restore_state() #reset to current epoch
                    
                sm_multi_P_Train_mean=np.mean(sm_multi_P_Train, axis=-1)
                sm_multi_P_Train_err = np.array([confinterval(sm_multi_P_Train[g,:]) for g in range(sm_multi_P_Train.shape[0])])
                sm_best_P_Train_mean=np.mean(sm_best_P_Train, axis=-1)
                sm_best_P_Train_err = np.array([confinterval(sm_best_P_Train[g,:]) for g in range(sm_best_P_Train.shape[0])])
                
                sm_multi_P_Val_mean=np.mean(sm_multi_P_Val, axis=-1)
                sm_multi_P_Val_err = np.array([confinterval(sm_multi_P_Val[g,:]) for g in range(sm_multi_P_Val.shape[0])])
                sm_best_P_Val_mean=np.mean(sm_best_P_Val, axis=-1)
                sm_best_P_Val_err = np.array([confinterval(sm_best_P_Val[g,:]) for g in range(sm_best_P_Val.shape[0])])
                
                Y_Train = Y_Train.detach().cpu().numpy()
                Y_Val = Y_Val.detach().cpu().numpy()
                 
                 
                # current plot
                ax = plt.subplot(sp_grid[1,3])
                #lims=(-15,-4)
                #lims=(-18,-4)
                lims=(-25,-4)
                plt.ylim(lims[0], lims[1])
                plt.xlim(lims[0], lims[1])                
                ax.errorbar(Y_Train, sm_multi_P_Train_mean, yerr=sm_multi_P_Train_err, color=summary_train_clr, marker=".", alpha=0.4, zorder=1, ls='')
                if(validation_db_file is not None):
                    ax.errorbar(Y_Val, sm_multi_P_Val_mean, yerr=sm_multi_P_Val_err, color=summary_unkn_clr, marker=".", alpha=0.4, zorder=2, ls='')
                diag = np.linspace(lims[0],lims[1], 10)
                plt.plot(diag, diag, '--k', zorder=0)
                plt.vlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
                plt.hlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
                ax.set(xlabel='dG Sim. or Expt. (kcal/mol)', ylabel='dG Predicted (kcal/mol)', title="Current Epoch")
                
                # best plot
                ax = plt.subplot(sp_grid[1,4])
                #lims=(-15,-4)
                #lims=(-18,-4)
                lims=(-25,-4)
                plt.ylim(lims[0], lims[1])
                plt.xlim(lims[0], lims[1])
                ax.errorbar(Y_Train, sm_best_P_Train_mean, yerr=sm_best_P_Train_err, color=summary_train_clr, marker=".", alpha=0.4, zorder=1, ls='')
                if(validation_db_file is not None):
                    ax.errorbar(Y_Val, sm_best_P_Val_mean, yerr=sm_best_P_Val_err, color=summary_unkn_clr, marker=".", alpha=0.4, zorder=2, ls='')
                diag = np.linspace(lims[0],lims[1], 10)
                plt.plot(diag, diag, '--k', zorder=0)
                plt.vlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
                plt.hlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
                ax.set(xlabel='dG Sim. or Expt. (kcal/mol)', ylabel='dG Predicted (kcal/mol)', title="Best s.m. expt.Val. Epoch (cross)")
            

            fig.suptitle("Representation: {:<15s}  NN width: {:<4d}  NN depth: {:<4d} Normalized X: {}\nKnown ligs: {:<8d} final cross-validation RMSE: {:.3f}".format(
                         dr_name, hl_w, hl_depth, normalize_x, len(known_idxs), rec.RMSD_XVal[-1]), y=0.95)

            plt.tight_layout()
            if(save_plot):
                fig.savefig(save_plot)
                
            #plt.show();

            
            
        if ep%backup_models_every == backup_models_every-1:
            print(f"Epoch {ep}. Backing up models.", flush=True)
            os.makedirs(save_folder+"/training_models", exist_ok=True)
            for i_m, m in enumerate(models):
                torch.save(m.state_dict(), save_folder+f"/training_models/model_{i_m}_{settings_hash}_training.ptmod")
                m.restore_state() # flip to best state of the model
                torch.save(m.state_dict(), save_folder+f"/training_models/model_{i_m}_{settings_hash}_best_so_far.ptmod")
                m.restore_state() # restore current state
            if(train_summary_model):
                for i_m, summary_model in enumerate(summary_models):
                    torch.save(summary_model.state_dict(), save_folder+f"/training_models/summary_model_{i_m}_{settings_hash}_training.ptmod")
                    summary_model.restore_state() # flip to best state of the model
                    torch.save(summary_model.state_dict(), save_folder+f"/training_models/summary_model_{i_m}_{settings_hash}_best_so_far.ptmod")
                    summary_model.restore_state() # restore current state
                
            pickle.dump( (settings, metrics), open(save_folder+f"/training_models/metrics_{settings_hash}_training.pickle", "wb" ) )
            #pickle.dump( (final_Pred_KFold), open(save_folder+f"/training_models/predictions_current_{settings_hash}_Ep{ep}_training.pickle", "wb" ) )
            
            
            if(predict_all_ligs):
                best_Pred_KFold=[]
                for m in models:
                    m.restore_state() # flip to best state of the model
                    best_P, best_Y = m.get_predictions_from_batches(all_generator, with_np=False)
                    m.restore_state() # restore current state
                    best_Pred_KFold.append(best_P.detach().cpu().numpy()[:,0])
                best_Pred_KFold = np.vstack(best_Pred_KFold)
                best_Pred_KFold = ( np.mean(best_Pred_KFold, axis=0),  np.std(best_Pred_KFold, axis=0)/np.sqrt(XVal_blocks))
                multi_model_record.best_pred=best_Pred_KFold
                pickle.dump( (best_Pred_KFold), open(save_folder+f"/training_models/predictions_best_{settings_hash}_Ep{ep}_training.pickle", "wb" ) )
                
                if(train_summary_model):
                    best_Pred_sm=[]
                    for summary_model in summary_models:
                        summary_model.restore_state()
                        best_P, best_Y = summary_model.get_predictions_from_batches(all_generator, with_np=False)
                        summary_model.restore_state()
                        best_Pred_sm.append(best_P.detach().cpu().numpy()[:,0])
                    best_Pred_sm = np.vstack(best_Pred_sm)
                    best_Pred_sm = ( np.mean(best_Pred_sm, axis=0),  np.std(best_Pred_sm, axis=0)/np.sqrt(XVal_blocks))
                    multi_model_record.summary_model_best_pred = best_Pred_sm
                    pickle.dump( best_Pred_sm, open(save_folder+f"/training_models/summary_model_predictions_best_{settings_hash}_Ep{ep}_training.pickle", "wb" ) )
            else:
                multi_model_record.best_pred=None

    #save the models
    print("Training done. Saving models.", flush=True)
    os.makedirs(save_folder, exist_ok=True)
    # torch.save(summary_model.state_dict(), save_folder+"/summary_model.ptmod")
    for i_m, m in enumerate(models):
        torch.save(m.state_dict(), save_folder+f"/model_{i_m}_{settings_hash}_ep_{n_Epochs}.ptmod")
        
    if(predict_all_ligs):
        best_Pred_KFold=[]
        for m in models:
            m.restore_state() # flip to best state of the model
            best_P, best_Y = m.get_predictions_from_batches(all_generator, with_np=False)
            m.restore_state() # restore current state
            best_Pred_KFold.append(best_P.detach().cpu().numpy()[:,0])
        best_Pred_KFold = np.vstack(best_Pred_KFold)
        multi_model_record.best_pred=( np.mean(best_Pred_KFold, axis=0),  np.std(best_Pred_KFold, axis=0)/np.sqrt(XVal_blocks))
        if(train_summary_model):
            best_Pred_sm=[]
            for summary_model in summary_models:
                summary_model.restore_state()
                best_P, best_Y = summary_model.get_predictions_from_batches(all_generator, with_np=False)
                summary_model.restore_state()
                best_Pred_sm.append(best_P.detach().cpu().numpy()[:,0])
            best_Pred_sm = np.vstack(best_Pred_sm)
            best_Pred_sm = ( np.mean(best_Pred_sm, axis=0),  np.std(best_Pred_sm, axis=0)/np.sqrt(XVal_blocks))
            multi_model_record.summary_model_best_pred = best_Pred_sm
    else:
        multi_model_record.best_pred=None
        if(train_summary_model):
            multi_model_record.summary_model_best_pred=None
    
    if(predict_all_ligs):
        final_Pred_KFold=[]
        for m in models:
            final_P, final_Y = m.get_predictions_from_batches(all_generator, with_np=False)
            final_Pred_KFold.append(final_P.detach().cpu().numpy()[:,0])
        final_Pred_KFold = np.vstack(final_Pred_KFold)
        final_Pred_KFold = ( np.mean(final_Pred_KFold, axis=0),  np.std(final_Pred_KFold, axis=0)/np.sqrt(XVal_blocks))
        multi_model_record.final_pred = final_Pred_KFold
        if(train_summary_model):
            final_Pred_sm=[]
            for summary_model in summary_models:
                final_P, final_Y = summary_model.get_predictions_from_batches(all_generator, with_np=False)
                final_Pred_sm.append(final_P.detach().cpu().numpy()[:,0])
            final_Pred_sm = np.vstack(final_Pred_sm)
            final_Pred_sm = ( np.mean(final_Pred_sm, axis=0),  np.std(final_Pred_sm, axis=0)/np.sqrt(XVal_blocks))
            multi_model_record.summary_model_final_pred = final_Pred_sm
    else:
        multi_model_record.final_pred=None
        if(train_summary_model):
            multi_model_record.summary_model_final_pred=None
    
    metrics=multi_model_record
    pickle.dump( (settings, metrics), open(settings_fname, "wb" ) )
    
    #temp_dir.cleanup()
    
    





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
    
    parser = argparse.ArgumentParser(description='Builds NN models for dG with cross validation.\nThis version uses ensemble models for both K-folds and summary_model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', dest='settings_fname', type=str, nargs='+',
                       help='settings pickle file')
    parser.add_argument('--redo', dest='redo', type=str2bool, nargs='?', const=True, default=False,
                        help="Recalculate even if output already exists.")
    parser.add_argument('-v', dest='show_progress', action='store_true', help="show progress.")
    parser.add_argument('--cpu', dest='cpu_only', action='store_true', help="don't use GPU.")
    parser.add_argument('--datafolder', dest='datafolder', type=str,
                        default="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/prediction_step_0",
                       help='where the dataset is stored')
    
    parser.add_argument('--save_folder_base', dest='save_folder_base',
                default=f"{pocket_fit_folder}/prediction_step_2/meta_param_search",
                help="path to save models in")
    parser.add_argument('--ligsf', dest='ligsf',
                default=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_with_2D_no_core_tSNE_and_step2_dG.pickle",
                help="Pickle file with full ligands and their simulated dG values.")
    #parser.add_argument('--nocoref', dest='nocoref',
                #default=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_no_core_ligs_with_step2_dG.pickle",
                #help="Pickle file with nocore ligands.")
    parser.add_argument('--no_pred', dest='pred', action='store_false', help="don't predict dG of all the ligands in the dataset. "
                "Uses less memory and doesn't need all the ligands on the HDD if normalization cache already exists.")
    parser.add_argument('--sm', dest='sm', action='store_true', help="Also train the summary model.")
    
    parser.add_argument('--valsf', dest='valsf',
                default=None,
                help="Pickle file with full validation ligands and their experimental dG values.")
    parser.add_argument('--nocorevalsf', dest='nocorevalsf',
                default=None,
                help="Pickle file with nocore validation ligands.")
    
    parser.add_argument('--training_lig_ids', dest='training_lig_ids', type=int, nargs='+',
                       help='indeces of training ligands')
    
    parser.add_argument('--n_loader_workers', dest='n_loader_workers', type=int,
                       help='number of threads used in a dataloader')
    
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
        GPU_ordinal="unknown"
        if('GPU_DEVICE_ORDINAL' in os.environ): # set by SLURM
            #GPU_ID_to_use = os.environ['GPU_DEVICE_ORDINAL']
            print(f"GPU_DEVICE_ORDINAL is set (to {os.environ['GPU_DEVICE_ORDINAL']})")
            print("We are on a SLURM cluster. This means the job will only see 1 GPU")
            GPU_ID_to_use=0
            GPU_ordinal=int(os.environ['GPU_DEVICE_ORDINAL'])
        elif("CUDA_VISIBLE_DEVICES" in os.environ): # SGE sets this instead
            print(f"CUDA_VISIBLE_DEVICES is set (to {os.environ['CUDA_VISIBLE_DEVICES']})")
            print("We are on an SGE cluster. This means the job will only see 1 GPU")
            GPU_ID_to_use=0
            GPU_ordinal=int(os.environ['CUDA_VISIBLE_DEVICES'])
        else: #as a fallback for other clusters
            GPU_ID_to_use = get_freer_gpu()

        dev = f"cuda:{GPU_ID_to_use}"
        print("Will use GPU id:", GPU_ID_to_use, "\twith a device ordinal:", GPU_ordinal)
        
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
        run=True
        while(run):
            try:
                train_model_with_settings_general_ensemble(f, args.redo, device, args.show_progress,
                            datafolder=args.datafolder, save_folder_base=args.save_folder_base,
                            all_ligs_db_file=args.ligsf,
                            #all_no_core_ligs_db_file=args.nocoref,
                            all_no_core_ligs_db_file=None,
                            validation_db_file=args.valsf, no_core_validation_db_file=args.nocorevalsf,
                            predict_all_ligs=args.pred, train_summary_model=args.sm,
                            training_ligs_idxs_in_all_ligs=args.training_lig_ids)
                                
                torch.cuda.empty_cache()
                run=False
            except Exception as e:
                sys.stderr.write(repr(e)+'\n')
                if(type(e) is type(RuntimeError()) and e.args[0]=="Nan in metrics"):
                    print("Nan detected in the loss or XVal_RMSE. Restrating training.")
                else:
                    print(f"Unknown exception: {e}. Terminating training of this model.")
                    import traceback
                    print(traceback.format_exc())
                    run=False
            
    exit(0);
