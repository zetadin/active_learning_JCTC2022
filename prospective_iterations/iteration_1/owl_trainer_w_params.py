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
folder=f"{pocket_fit_folder}/prediction_step_0/repr_scan_step_0_on_preliminary_calc_ddG"
step0_db_file=f"{pocket_fit_folder}/ddG_step_0_sigmahole_scaled_dummy_masses/stable_ligs.pickle"
all_ligs_db_file=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_with_2D_no_core_tSNE_and_step0_dG.pickle"
all_no_core_ligs_db_file=f"{pocket_fit_folder}/filtered_subset_without_weird_chemistry_no_core_ligs_with_step0_dG.pickle"

Bohr2Ang=0.529177249
RT=0.001985875*300 #kcal/mol

import sys
sys.path.insert(0, f"{folder}/..") 
sys.path.insert(0, folder)

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
        self.datasubsets=["Train", "Train_err", "XVal", "unknown"]
        
        for m in self.metrics:
            for d in self.datasubsets:
                setattr(self, m+'_'+d, [])
                
        self.kendalltau_XVal=[]
        self.final_pred=None
        self.best_pred=None


def train_model_with_settings(settings_fname, redo=False, device=torch.device("cpu"), show_progress=False,
                              datafolder="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/prediction_step_0"):
    print("\n\n\n")
    
    #load the settings
    with open(settings_fname, 'rb') as f:
        settings, metrics = pickle.load(f)
    
    if(metrics is not None and not redo):
        print(f"{settings_fname} is already trained. Will NOT retrain it.")
        return;
    
    representation_flags, dr_name, normalize_x, shuffle_seed, n_Epochs, hl_w, hl_depth, init_learning_rate, learning_rate_decay, weight_decay, impfilt, X_filter, weighted, shiftY, use_dropout  = settings
    
    print("Representation is:", dr_name, flush=True)
    
    #temp_dir = tempfile.TemporaryDirectory()
    #if()
    #print('Copied cached representation locally')
    
    # load set_0 known ligands and make dataloader
    with open(step0_db_file, 'rb') as f:
        ligs = pickle.load(f)
        
    with open(all_ligs_db_file, 'rb') as f:
        all_ligs = pickle.load(f)
        
    with open(all_no_core_ligs_db_file, 'rb') as f:
        no_core_all_ligs = pickle.load(f)
        
    lig_idxs_in_all=[]
    lig_idxs_in_known=[]
    known_lig_idxs=[l.GetProp('ID') for l in ligs]
    for i,lig in enumerate(all_ligs):
        if(lig.GetProp('ID') in known_lig_idxs):
            lig_idxs_in_all.append(i)
            lig_idxs_in_known.append(known_lig_idxs.index(lig.GetProp('ID')))
    
    no_core_ligs=[None]*len(ligs)
    for i in range(len(ligs)):
        no_core_ligs[lig_idxs_in_known[i]]=no_core_all_ligs[lig_idxs_in_all[i]]
        
    #ligs=[l for l in ligs if not("<" in l.GetProp("[Q] hPDE2_pIC50") or ">" in l.GetProp("[Q] hPDE2_pIC50"))]

    #DS=CustomMolModularDataset(ligs=ligs, representation_flags=representation_flags, normalize_x=True, datafolder=datafolder)
    #DS.find_normalization_factors()
    
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
    print("# starting ligands:",n_starting_ligs)
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
    else:
        weight_func=None
        
    if(use_dropout):
        p_dropout=np.repeat(0.5, hl_depth+1)
    else:
        p_dropout=np.repeat(0.0, hl_depth+1)
        
    noise=0.0
    
        

    train_clr="blue"
    Xval_clr="purple"
    unkn_clr="green"
    summary_train_clr="darkred"
    summary_unkn_clr="peru"


    #generator for all data
    full_dataset = CustomMolModularDataset(ligs=all_ligs, no_core_ligs=no_core_all_ligs,
                                            representation_flags=representation_flags, normalize_x=normalize_x,
                                            X_filter=X_filter, datafolder=datafolder)
    if(normalize_x):
        full_dataset.find_normalization_factors()
    all_generator = torch.utils.data.DataLoader(full_dataset, shuffle=False, batch_size=eval_batchsize)

            
    multi_model_record=None
    global_models=None
    global_summary_model=None
    
    
    dr_detailed_name=dr_name if not "ESP" in dr_name else dr_name+"_p{}_a{}".format(full_dataset.grid_padding, full_dataset.grid_spacing)
    
    
    sha = hashlib.sha256()
    sha.update(pickle.dumps(settings))
    settings_hash=sha.hexdigest()[:10]
        
    save_folder=folder+f"/meta_param_search/models/{dr_name}/"
    save_folder+=f"known_{len(known_idxs)}_width_{hl_w}_depth_{hl_depth}"+\
                    f"_trd_{learning_rate_decay}_wd_{weight_decay}"+("" if not impfilt else f"_impfilt_{impfilt}")+\
                    ('' if not normalize_x else '_norm_X')+("" if not weighted else "_weightedbyY")+("" if not use_dropout else f"_dropout{p_dropout[0]}")+\
                    ('' if noise<=0.0 else f"_noise_{noise}")+("_shiftY" if shiftY else "")+f"_{n_Epochs}ep"
    os.makedirs(save_folder, exist_ok=True)
    save_plot=save_folder+f"/summary_{settings_hash}_ep_{n_Epochs}.png"
   
   
    #################
    # do the training
    #################
    
    n_features = full_dataset[0][0].shape[-1]
    known_Y=allYdata[known_idxs]

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

        Train_set = CustomMolModularDataset([ligs[idx] for idx in Train_indeces], no_core_ligs=[no_core_ligs[idx] for idx in Train_indeces],
                                            representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter, datafolder=datafolder)
        XVal_set  = CustomMolModularDataset([ligs[idx] for idx in XVal_indeces ], no_core_ligs=[no_core_ligs[idx] for idx in Train_indeces],
                                            representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter, datafolder=datafolder)
        Train_set.copy_normalization_factors(full_dataset)
        XVal_set.copy_normalization_factors(full_dataset)
        
        #build_internal_filtered_cache
        Train_set.build_internal_filtered_cache()
        XVal_set.build_internal_filtered_cache()
        
        #create training, cross validation, and test datagens
        model.training_generator            = torch.utils.data.DataLoader(Train_set, shuffle=True,  batch_size=batchsize)
        model.training_generator_no_shuffle = torch.utils.data.DataLoader(Train_set, shuffle=False, batch_size=eval_batchsize)
        model.crossvalidation_generator     = torch.utils.data.DataLoader(XVal_set,  shuffle=False, batch_size=eval_batchsize)
        
        del Train_set, XVal_set
        
        models.append(model)

    multi_model_record = MultiTrainingRecord_owl()

    print("Training for", n_Epochs, "epochs", flush=True)
    myrange=range(0, n_Epochs)
    #if(show_progress):
        #myrange=trange(0, n_Epochs, desc="Epoch")
    #for ep in trange(0, n_Epochs, desc="Epoch"):
    #for ep in range(0, n_Epochs):
    for ep in myrange:
        #train K_fold models
        for drop in range(XVal_blocks):

            model = models[drop]
            model.train_epoch(eval_every=eval_every)

        if ep%eval_every == eval_every-1 or ep == 0:
            all_Y_known=torch.zeros((len(known_idxs), 1), dtype=torch.float, device=device)
            multi_P_XVal=torch.zeros((len(known_idxs), 1), dtype=torch.float, device=device)
#             multi_P_eval=torch.zeros((len(evaluation_half), 1, XVal_blocks), dtype=torch.float, device=device)
            for drop in range(XVal_blocks):
                test_ndxs=model_XVal_idxs[drop]
                P_XVal, Y_XVal = models[drop].get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                multi_P_XVal[test_ndxs,:]= P_XVal
                all_Y_known[test_ndxs,:] = Y_XVal
                        
            np_all_Y_known=all_Y_known.cpu().detach().numpy()[:,0]
            np_multi_P_XVal=multi_P_XVal.cpu().detach().numpy()[:,0]
                
            XVal_loss=models[0].loss_fn(all_Y_known, multi_P_XVal).cpu().detach().tolist()
            XVal_RMSE=torch.sqrt(models[0].loss_mse(all_Y_known, multi_P_XVal)).cpu().detach().numpy()
            XVal_Cor=np.corrcoef(np_all_Y_known.flatten(), np_multi_P_XVal.flatten())[0,1]
            XVal_KT=sp.stats.kendalltau(np_all_Y_known.flatten(), np_multi_P_XVal.flatten())[0]
            
            
            #true and false positive rates + AUC
            XVal_FP, XVal_TP, XVal_AUC = get_FPR_TPR_AUC(np_all_Y_known, np_multi_P_XVal, high_binder_cutoff)
            
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
                if ("_err" in d):
                    srcd=d[:-4]
                    f=func_err
                
                for metric in multi_model_record.metrics:
                    atrname=metric+'_'+d
                    srcname=metric+'_'+srcd
                    getattr(multi_model_record, atrname).append(f(srcname))
                    
            if(show_progress):
                print(f"\tEpoch: {ep}\t train_loss={multi_model_record.loss_Train[-1]}\t XVal_RMSE={XVal_RMSE}", flush=True)

        #plot all models at once
        if ep%plot_every == plot_every-1:

            fig=plt.figure(figsize=(20,8))
            sp_grid = gridspec.GridSpec(2, 5)

            ax = plt.subplot(sp_grid[0,0])
            plt.ylim(0, 2.5)
            rec = multi_model_record
            bi=np.argmin(rec.loss_XVal)

            ax.errorbar(rec.epoch, rec.loss_Train, yerr=rec.loss_Train_err, color=train_clr, alpha=0.3)
            ax.plot(rec.epoch, rec.loss_Train, alpha=0.3, label="training", color=train_clr)
            ax.plot(rec.epoch, rec.loss_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.scatter(rec.epoch[bi], rec.loss_Train[bi], color=train_clr, marker="*")
            ax.scatter(rec.epoch[bi], rec.loss_XVal[bi], color=Xval_clr, marker="*")
            eprange=[0,ep]
            MAE_best=[rec.loss_XVal[bi],rec.loss_XVal[bi]]
            MAE_last=[rec.loss_XVal[-1],rec.loss_XVal[-1]]
            ax.plot(eprange, MAE_best, '--k', zorder=0, alpha=0.3, label=None)
            ax.plot(eprange, MAE_last, '--k', zorder=0, alpha=0.3, label=None)

            ax.set(xlabel='Epoch', ylabel='Loss (MAE)')

            ax = plt.subplot(sp_grid[0,1])
            plt.ylim(0, 2.5)
            ax.errorbar(rec.epoch, rec.RMSD_Train, yerr=rec.RMSD_Train_err, color=train_clr, alpha=0.3)
            ax.plot(rec.epoch, rec.RMSD_Train, alpha=0.3, label="training", color=train_clr)
            ax.plot(rec.epoch, rec.RMSD_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.scatter(rec.epoch[bi], rec.RMSD_Train[bi], color=train_clr, marker="*")
            ax.scatter(rec.epoch[bi], rec.RMSD_XVal[bi], color=Xval_clr, marker="*")
            ax.set(xlabel='Epoch', ylabel='RMSD')

            ax = plt.subplot(sp_grid[0,2])
            plt.ylim(0, 1)
            ax.errorbar(rec.epoch, rec.Cor_Train, yerr=rec.Cor_Train_err, color=train_clr, alpha=0.3)
            ax.plot(rec.epoch, rec.Cor_Train, alpha=0.3, label="training", color=train_clr)
            ax.plot(rec.epoch, rec.Cor_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.scatter(rec.epoch[bi], rec.Cor_Train[bi], color=train_clr, marker="*")
            ax.scatter(rec.epoch[bi], rec.Cor_XVal[bi], color=Xval_clr, marker="*")
            ax.set(xlabel='Epoch', ylabel='Corelation')

            plt.legend()
            
            ################
            ax = plt.subplot(sp_grid[1,0])
            plt.ylim(0, 1)
            ax.plot(rec.epoch, rec.TP_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.errorbar(rec.epoch, rec.TP_Train, yerr=rec.TP_Train_err, label="training", color=train_clr, alpha=0.3)
            ax.set(xlabel='Epoch', ylabel='True Positive Rate')
            
            ax = plt.subplot(sp_grid[1,1])
            plt.ylim(0, 1)
            ax.plot(rec.epoch, rec.FP_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.errorbar(rec.epoch, rec.FP_Train, yerr=rec.FP_Train_err, label="training", color=train_clr, alpha=0.3)
            ax.set(xlabel='Epoch', ylabel='False Positive Rate')
            
            ax = plt.subplot(sp_grid[1,2])
            plt.ylim(0, 1)
            ax.plot(rec.epoch, rec.AUC_XVal, alpha=0.3, label="cross validation", color=Xval_clr)
            ax.errorbar(rec.epoch, rec.AUC_Train, yerr=rec.AUC_Train_err, label="training", color=train_clr, alpha=0.3)
            ax.set(xlabel='Epoch', ylabel='Area under ROC curve')
            
            

            #prepare data across multiple models            
            multi_P_XVal=torch.zeros(all_Y_known.shape, dtype=torch.float)
            best_P_XVal=torch.zeros(all_Y_known.shape, dtype=torch.float)
            multi_P_Train = np.zeros((all_Y_known.shape[0], XVal_blocks-1)) #indexed by [data, repeat]
            best_P_Train  = np.zeros((all_Y_known.shape[0], XVal_blocks-1)) #indexed by [data, repeat]

            for drop in range(XVal_blocks): #go through each block
                test_ndxs=model_XVal_idxs[drop]

                P_XVal, Y_XVal = models[drop].get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                multi_P_XVal[test_ndxs,:]=P_XVal.cpu().detach()
            
                models[drop].restore_state() #best epoch so far
                P_XVal, Y_XVal  = models[drop].get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                models[drop].restore_state() #reset to current epoch
                best_P_XVal[test_ndxs,:]=P_XVal.cpu().detach()



                #for training predictions,
                #go through each model
                for k, m in enumerate(models):               
                    j=k # j<drop
                    if(k==drop):
                        continue;
                    elif(k>drop):
                        j=k-1

                    P_train, Y_train = m.get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                    multi_P_Train[test_ndxs, j] = P_train.cpu().detach().numpy()[:,0]

                    m.restore_state() #best epoch so far
                    P_train, Y_train = m.get_predictions_from_batches(models[drop].crossvalidation_generator, with_np=False)
                    m.restore_state() #reset to current epoch
                    best_P_Train[test_ndxs, j] = P_train.cpu().detach().numpy()[:,0]
                    
          
            
            multi_P_XVal = multi_P_XVal.numpy()[:,0] #convert to numpy, ordered like known_Y
            multi_P_Train_mean= np.mean(multi_P_Train, axis=-1) # ordered like known_Y
            multi_P_Train_err = [confinterval(multi_P_Train[g,:]) for g in range(multi_P_Train.shape[0])]

            best_P_XVal = best_P_XVal.numpy()[:,0] #convert to numpy, ordered like known_Y
            best_P_Train_mean = np.mean(best_P_Train, axis=-1) # ordered like known_Y
            best_P_Train_err = [confinterval(best_P_Train[g,:]) for g in range(best_P_Train.shape[0])]

            #plot data
            ax = plt.subplot(sp_grid[0,3])
            #lims=(-15,-4)
            lims=(-18,-4)
            plt.ylim(lims[0], lims[1])
            plt.xlim(lims[0], lims[1])
            ax.errorbar(known_Y, multi_P_Train_mean, yerr=multi_P_Train_err, color=train_clr, marker=".", ls='', alpha=0.2, zorder=1)
            ax.scatter(known_Y, multi_P_XVal, color=Xval_clr, marker=".", alpha=0.4, zorder=2)
            #ligands selected for next traiing set:
            final_Pred_KFold=[]
            for m in models:
                final_P, final_Y = m.get_predictions_from_batches(all_generator, with_np=False)
                final_Pred_KFold.append(final_P.cpu().detach().numpy()[:,0])
            final_Pred_KFold = np.vstack(final_Pred_KFold)
            final_Pred_KFold = ( np.mean(final_Pred_KFold, axis=0),  np.std(final_Pred_KFold, axis=0)/np.sqrt(XVal_blocks))
            multi_model_record.final_pred = final_Pred_KFold
            #sorted_P_idxs = np.argsort(final_Pred_KFold[0])
            
            diag = np.linspace(lims[0],lims[1], 10)
            plt.plot(diag, diag, '--k', zorder=0)
            plt.vlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            plt.hlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            ax.set(xlabel='dG Experiment (kcal/mol)', ylabel='dG Predicted (kcal/mol)', title="Current Epoch")

            #plot best data
            ax = plt.subplot(sp_grid[0,4])
            plt.ylim(lims[0], lims[1])
            plt.xlim(lims[0], lims[1])
            ax.errorbar(known_Y, best_P_Train_mean, yerr=best_P_Train_err, color=train_clr, marker=".", ls='', alpha=0.2, zorder=1)
            ax.scatter(known_Y, best_P_XVal, color=Xval_clr, marker=".", alpha=0.4, zorder=2)
            diag = np.linspace(lims[0],lims[1], 10)
            plt.plot(diag, diag, '--k', zorder=0)
            plt.vlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            plt.hlines(high_binder_cutoff, lims[0], lims[1], colors='k', linestyles='dotted', label='')
            ax.set(xlabel='dG Experiment (kcal/mol)', ylabel='dG Predicted (kcal/mol)', title="Best XVal. Epoch (star)")
            

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
            pickle.dump( (settings, metrics), open(save_folder+f"/training_models/metrics_{settings_hash}_training.pickle", "wb" ) )
            pickle.dump( (final_Pred_KFold), open(save_folder+f"/training_models/predictions_current_{settings_hash}_Ep{ep}_training.pickle", "wb" ) )
            
            best_Pred_KFold=[]
            for m in models:
                m.restore_state() # flip to best state of the model
                best_P, best_Y = m.get_predictions_from_batches(all_generator, with_np=False)
                m.restore_state() # restore current state
                best_Pred_KFold.append(best_P.cpu().detach().numpy()[:,0])
            best_Pred_KFold = np.vstack(best_Pred_KFold)
            best_Pred_KFold = ( np.mean(best_Pred_KFold, axis=0),  np.std(best_Pred_KFold, axis=0)/np.sqrt(XVal_blocks))
            multi_model_record.best_pred=best_Pred_KFold
            pickle.dump( (best_Pred_KFold), open(save_folder+f"/training_models/predictions_best_{settings_hash}_Ep{ep}_training.pickle", "wb" ) )

    #save the models
    print("Training done. Saving models.", flush=True)
    os.makedirs(save_folder, exist_ok=True)
    # torch.save(summary_model.state_dict(), save_folder+"/summary_model.ptmod")
    for i_m, m in enumerate(models):
        torch.save(m.state_dict(), save_folder+f"/model_{i_m}_{settings_hash}_ep_{n_Epochs}.ptmod")
        
    best_Pred_KFold=[]
    for m in models:
        m.restore_state() # flip to best state of the model
        best_P, best_Y = m.get_predictions_from_batches(all_generator, with_np=False)
        m.restore_state() # restore current state
        best_Pred_KFold.append(best_P.cpu().detach().numpy()[:,0])
    best_Pred_KFold = np.vstack(best_Pred_KFold)
    multi_model_record.best_pred=( np.mean(best_Pred_KFold, axis=0),  np.std(best_Pred_KFold, axis=0)/np.sqrt(XVal_blocks))
    
    final_Pred_KFold=[]
    for m in models:
        final_P, final_Y = m.get_predictions_from_batches(all_generator, with_np=False)
        final_Pred_KFold.append(final_P.cpu().detach().numpy()[:,0])
    final_Pred_KFold = np.vstack(final_Pred_KFold)
    final_Pred_KFold = ( np.mean(final_Pred_KFold, axis=0),  np.std(final_Pred_KFold, axis=0)/np.sqrt(XVal_blocks))
    multi_model_record.final_pred = final_Pred_KFold
    
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
    
    parser = argparse.ArgumentParser(description='Builds NN models for dG with cross validation.',
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

    n_cpus = int(os.environ['NSLOTS'])
    torch.set_num_threads(n_cpus)
    print("Will use this many CPU threads:", n_cpus)
    ##################################################################################



    for f in args.settings_fname:
        try:
            train_model_with_settings(f, args.redo, device, args.show_progress, datafolder=args.datafolder)
            torch.cuda.empty_cache()
        except Exception as e:
            sys.stderr.write(repr(e)+'\n')
            
    exit(0);
