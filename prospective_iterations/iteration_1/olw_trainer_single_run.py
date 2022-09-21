from tqdm import tqdm, trange
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import scipy as sp
import time
import sys
import importlib
import os

from IPython import display
from IPython.display import clear_output
import copy
from copy import deepcopy
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

try:
    import cPickle as pickle
except:
    import pickle
    
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

folder="/home/ykhalak/Projects/ML_dG/pde2_dG/how_do_ligs_fit_in_pocket/adaptive_learning_test_from_morphed_structs/"
global_molecular_db_file=folder+"/../processed_ligs_w_morphing_sim_annealing_only_sucessfull.pickle"

Bohr2Ang=0.529177249
RT=0.001985875*300 #kcal/mol


import utils
from utils import *


##################################################################################
print("is CUDA available:", torch.cuda.is_available())

def get_freer_gpu():
    cmd="nvidia-smi -q -d UTILIZATION | grep -A5 GPU | grep 'Gpu'"
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    res = ps.communicate()[0].decode('utf-8').splitlines()
    gpu_load = [int(x.split()[-2]) for x in res]
    print("GPU loads are:",gpu_load, "%")
    return np.argmin(gpu_load)


if torch.cuda.is_available():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"CUDA_VISIBLE_DEVICES is set (to {os.environ['CUDA_VISIBLE_DEVICES']})")
        print("This means the job will only see 1 GPU")
        GPU_ID_to_use=0
    else:
        GPU_ID_to_use = get_freer_gpu()
    dev = "cuda:"+str(GPU_ID_to_use)
    print("Will use GPU id:", GPU_ID_to_use)
else:  
    dev = "cpu"
    print("Will use CPU only")
device = torch.device(dev)

n_cpus = int(os.environ['NSLOTS'])
torch.set_num_threads(n_cpus)
print("Will use this many CPU threads:", n_cpus)
##################################################################################



#build NN class
import NNs
from NNs import *

#build data loader
import custom_dataset_modular
from custom_dataset_modular import dataBlocks, CustomMolModularDataset

base_flags=[dataBlocks.MACCS, dataBlocks.Descriptors, dataBlocks.EState_FP, dataBlocks.Graph_desc, dataBlocks.Pharmacophore_feature_map]
extras_flags=[dataBlocks.MOE, dataBlocks.MQN, dataBlocks.GETAWAY, dataBlocks.AUTOCORR2D,
              dataBlocks.AUTOCORR3D, dataBlocks.BCUT2D, dataBlocks.WHIM, dataBlocks.RDF,
              dataBlocks.USR, dataBlocks.USRCUT, dataBlocks.PEOE_VSA, dataBlocks.SMR_VSA,
              dataBlocks.SlogP_VSA, dataBlocks.MORSE]
flags_2D=[dataBlocks.MACCS, dataBlocks.Descriptors, dataBlocks.Graph_desc, dataBlocks.BCUT2D]
flags_3D=[dataBlocks.EState_FP, dataBlocks.Pharmacophore_feature_map,
          dataBlocks.MOE, dataBlocks.MQN, dataBlocks.GETAWAY, dataBlocks.AUTOCORR2D,
          dataBlocks.AUTOCORR3D, dataBlocks.WHIM, dataBlocks.RDF,
          dataBlocks.USR, dataBlocks.USRCUT, dataBlocks.PEOE_VSA, dataBlocks.SMR_VSA,
          dataBlocks.SlogP_VSA, dataBlocks.MORSE]


representation_flags=[0]*len(dataBlocks)
# for b in base_flags:
#     representation_flags[int(b)]=1
# for b in extras_flags:
#     representation_flags[int(b)]=1
# for b in flags_2D:
#     representation_flags[int(b)]=1
# for b in flags_3D:
#     representation_flags[int(b)]=1
# representation_flags[int(dataBlocks.atom_hot)]=1
#representation_flags[int(dataBlocks.ESP_full_grid)]=1
#representation_flags[int(dataBlocks.ESP_on_vdw_surf)]=1

#representation_flags[int(dataBlocks.PLEC)]=1
representation_flags[int(dataBlocks.PLEC_filtered)]=1


# dr_name=""
# for i in range(len(representation_flags)):
#     if representation_flags[i]:
#         dr_name+=dataBlocks(i).name+'_'
# dr_name=dr_name[:-1]

dr_name="PLEC"
print("Representation is:", dr_name)


# load ligands and make dataloader
with open(global_molecular_db_file, 'rb') as f:
    ligs = pickle.load(f)
ligs=[l for l in ligs if not("<" in l.GetProp("[Q] hPDE2_pIC50") or ">" in l.GetProp("[Q] hPDE2_pIC50"))]

DS=CustomMolModularDataset(ligs=ligs, representation_flags=representation_flags, normalize_x=True)
DS.find_normalization_factors()


#subdivide dataset
normalize_x=True
high_binder_cutoff=-5*np.log(10)

#XVal division function
XVal_blocks=5
from sklearn.model_selection import KFold
def data_split(data):
    kf=KFold(n_splits=XVal_blocks, shuffle=True, random_state=12345)
    return(kf.split(data))

allYdata=-RT*(np.array([float(lig.GetProp('[V] hPDE2_pIC50')) for lig in ligs]))*np.log(10)
minY=np.min(allYdata)
maxY=np.max(allYdata)

    #shuffle ligand indeces
all_idxs_shuffled=np.arange(len(ligs))
np.random.seed(12345678)
np.random.shuffle(all_idxs_shuffled) # in place transform

    #halfNhalf
good_binders=np.array([i for i in all_idxs_shuffled if allYdata[i]<high_binder_cutoff], dtype=int)
bad_binders=np.array([i for i in all_idxs_shuffled if allYdata[i]>=high_binder_cutoff], dtype=int)
training_half=all_idxs_shuffled
evaluation_half=np.array([])

    #shuffle training & eval data
np.random.shuffle(training_half)
np.random.shuffle(evaluation_half)
                                                                    

    #extract data "known" at the current loop
n_starting_ligs=min(5000, len(training_half))
print("# starting ligands:",n_starting_ligs)
known_idxs=training_half[:n_starting_ligs]
unknown_idxs=training_half[n_starting_ligs:]







#########################################################################
############################ Train models ###############################
#########################################################################

#n_Epochs=20000
#n_Epochs=200
n_Epochs=2000
#hl_w=10
#hl_w = 120
#hl_w = 480
hl_w=40
hl_depth=2
#batchsize=50
batchsize=500
eval_batchsize=200
activation="relu"
init_learning_rate=5e-3
learning_rate_decay=10000 #order of magnitude in this many epochs
#learning_rate_decay=0
weight_decay=1e-3
#weight_decay=0

#eval_every=20

eval_every=10
#plot_every=n_Epochs
plot_every=100

expand_by=100

normalize_x=True
X_filter=None
impfilt=None

#impfilt=0.0075
#impfilt=0.01
#X_filter=folder+"/models/MACCS_w_desc2featmap_ESP_full_grid_w_atom_hot/known_1044_width_120_depth_2_trd_0_wd_0.001_norm_X/"+f"/Xfilter_importance_over_{impfilt}.pickle"
#X_filter="/home/ykhalak/Projects/ML_dG/pde2_dG/how_do_ligs_fit_in_pocket/adaptive_learning_test_from_morphed_structs/models/MACCS_w_desc2featmap_w_extras/known_2088_width_120_depth_2_trd_10000_wd_0.001_norm_X_weightedbyY_dropout0.5_shiftY/"+f"/Xfilter_importance_over_{impfilt}.pickle"

# h="82602"
# impfilt=f"hash_{h}"
# combined_filter_folder=folder+"models/MACCS_w_desc2featmap_ESP_full_grid_w_atom_hot_w_extras/"
# X_filter=combined_filter_folder+f"/Xfilter_combined_hash_{h}.pickle"
# if(not os.path.exists(X_filter)):
#     raise(Exception(f"No such file: {X_filter}"))


weighted=True
if(weighted):
    weight_func=get_weights
else:
    weight_func=None
    
use_dropout=True
if(use_dropout):
    p_dropout=np.repeat(0.5, hl_depth+1)
else:
    p_dropout=np.repeat(0.0, hl_depth+1)
    
noise=0.0
shiftY=True
    
    
    

train_clr="blue"
Xval_clr="purple"
unkn_clr="green"
summary_train_clr="darkred"
summary_unkn_clr="peru"


#generator for all data
full_dataset = CustomMolModularDataset(ligs=ligs, representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter)
if(normalize_x):
    full_dataset.find_normalization_factors()
all_generator = torch.utils.data.DataLoader(full_dataset, shuffle=False, batch_size=eval_batchsize)


class MultiTrainingRecord:
    def __init__(self):
        self.epoch=[]
        
        self.metrics=["loss", "RMSD", "Cor", "TP", "FP", "AUC"]
        self.datasubsets=["Train", "Train_err", "XVal", "unknown", #"unknown_err",
                          "summary_model_Train", "summary_model_unknown"]
        
        for m in self.metrics:
            for d in self.datasubsets:
                setattr(self, m+'_'+d, [])
        
multi_model_record=None
global_models=None
global_summary_model=None




def train_everything(save_plot=None):
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

        Train_set = CustomMolModularDataset([ligs[idx] for idx in Train_indeces], representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter)
        XVal_set  = CustomMolModularDataset([ligs[idx] for idx in XVal_indeces ], representation_flags=representation_flags, normalize_x=normalize_x, X_filter=X_filter)
        Train_set.copy_normalization_factors(full_dataset)
        XVal_set.copy_normalization_factors(full_dataset)
        
        #create training, cross validation, and test datagens
        model.training_generator            = torch.utils.data.DataLoader(Train_set, shuffle=True,  batch_size=batchsize)
        model.training_generator_no_shuffle = torch.utils.data.DataLoader(Train_set, shuffle=False, batch_size=eval_batchsize)
        model.crossvalidation_generator     = torch.utils.data.DataLoader(XVal_set,  shuffle=False, batch_size=eval_batchsize)
        
        del Train_set, XVal_set
        
        models.append(model)

    multi_model_record = MultiTrainingRecord()

    for ep in trange(0, n_Epochs, desc="Epoch"):
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

        #plot all models at once
        if ep%plot_every == plot_every-1:
            clear_output(wait=True)

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
            lims=(-15,-4)
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
            final_Pred_KFold = np.mean(final_Pred_KFold, axis=0)
            sorted_P_idxs = np.argsort(final_Pred_KFold)
            
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
            

            fig.suptitle("Representation: {:<15s}  NN width: {:<4d}  NN depth: {:<4d} Normalized X: {}\nKnown ligs: {:<8d} Evaluation ligs: {:<8d} final cross-validation RMSE: {:.3f}".format(
                         dr_name, hl_w, hl_depth, normalize_x, len(known_idxs), len(evaluation_half), rec.RMSD_XVal[-1]), y=0.95)

            plt.tight_layout()
            if(save_plot):
                fig.savefig(save_plot)
            plt.show();
            
            print("Pearson corr.:\t", np.corrcoef(known_Y, best_P_XVal)[0,1])
            print("Kendall-tau corr.:\t", sp.stats.kendalltau(known_Y, best_P_XVal)[0])
            

    
    #final predictions across whole dataset   
    final_Pred_KFold=[]
    for m in models:
        final_P, final_Y = m.get_predictions_from_batches(all_generator, with_np=False)
        final_Pred_KFold.append(final_P.cpu().detach().numpy()[:,0])
    final_Pred_KFold = np.vstack(final_Pred_KFold)
    final_Pred_KFold = np.mean(final_Pred_KFold, axis=0)
    
    final_Pred_summary_model=None
    
    #save the models
    save_folder=f"{folder}/models/{dr_name}/known_{len(known_idxs)}_width_{hl_w}_depth_{hl_depth}"+\
                f"_trd_{learning_rate_decay}_wd_{weight_decay}"+("" if not impfilt else f"_impfilt_{impfilt}")+\
                ('' if not normalize_x else '_norm_X')+("" if not weighted else "_weightedbyY")+("" if not use_dropout else f"_dropout{p_dropout[0]}")+\
                ('' if noise<=0.0 else f"_noise_{noise}")+("_shiftY" if shiftY else "")
    os.makedirs(save_folder, exist_ok=True)
#     torch.save(summary_model.state_dict(), save_folder+"/summary_model.ptmod")
    for i_m, m in enumerate(models):
        torch.save(m.state_dict(), save_folder+f"/model_{i_m}.ptmod")
    
    
    return(multi_model_record, final_Pred_KFold, final_Pred_summary_model)

#raise()
    
dr_detailed_name=dr_name if not "ESP" in dr_name else dr_name+"_p{}_a{}".format(full_dataset.grid_padding, full_dataset.grid_spacing)
#dr_detailed_name="only_cluster_11_"+(dr_name if not "ESP" in dr_name else dr_name+"_p{}_a{}".format(full_dataset.grid_padding, full_dataset.grid_spacing))
rec, final_Pred_KFold, final_Pred_summary_model, = train_everything(save_plot=folder+"/new_end_plots"+"/{}_known_{}_width_{}_depth_{}_trd_{}_wd_{}{}{}{}{}{}{}_{}ep.png".format(
                                                                        dr_detailed_name, len(known_idxs), hl_w, hl_depth, learning_rate_decay,
                                                                        weight_decay, "" if not normalize_x else "_norm_X",
                                                                        "" if not impfilt else f"_impfilt_{impfilt}",
                                                                        "" if not weighted else "_weightedbyY",
                                                                        "" if not use_dropout else f"_dropout{p_dropout[0]}",
                                                                        "" if noise<=0.0 else f"_noise_{noise}",
                                                                        "_shiftY" if shiftY else "",
                                                                        n_Epochs))
