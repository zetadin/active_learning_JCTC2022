import numpy as np
import scipy as sp
import scipy.stats as st
import sklearn
from rdkit import Chem
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem import AllChem, Draw, Descriptors, rdmolfiles, rdMolAlign, rdmolops, rdchem, rdMolDescriptors, ChemicalFeatures
#from sklearn.metrics import roc_auc_score, roc_curve

#turn Y and P data into high binder probabilities for ROC curve calculation
def probabilitize(P,Y):
    #high binders have negative values of Y & P

    R=1-((P-np.min(Y))/(np.max(Y)-np.min(Y))) #1- is to flip so that high binders have low P
    R[R>1]=1
    R[R<0]=0
    return(R)

#find false and true positive rates
def get_FPR_and_TPR(Y,P, cut):
    True_Y_ndx=np.argwhere(Y<cut)
    filt_P=P[True_Y_ndx]
    Num_TP=filt_P[filt_P<cut].shape[0]
    if(Y[Y<cut].shape[0]==0):
        TPR=0
    else:
        TPR=Num_TP/Y[Y<cut].shape[0]

    filt_P=P[P<cut] #num pos P
    Num_FP=filt_P.shape[0]-Num_TP
    if(Y[Y>=cut].shape[0]==0):
        FPR=0
    else:
        FPR=Num_FP/Y[Y>=cut].shape[0]

    return(FPR, TPR)


def get_precision(Y,P, cut):
    positive_ndx=np.argwhere(P<cut)
    TP_ndx=np.argwhere(np.logical_and(Y<cut, P<cut))
    #filt_positive=P[positive_ndx]
    #filt_TP=P[TP_ndx]
    Num_TP=TP_ndx.shape[0]
    num_positive=positive_ndx.shape[0]
    if(num_positive==0):
        prec=0
    else:
        prec=Num_TP/num_positive
    return(prec)

def get_FPR_TPR_AUC(Y,P, cut):
    FP, TP = get_FPR_and_TPR(Y,P,cut)
    hb_prob_Y=np.where(Y<cut, 1, 0)
    #hb_prob_P=probabilitize(P,Y)
    hb_prob_P=np.where(P<cut, 1, 0)
    if (len(np.unique(hb_prob_Y)) != 2):
        AUC=np.nan
    else:
        AUC = sklearn.metrics.roc_auc_score(hb_prob_Y, hb_prob_P)
    return(FP, TP, AUC)

from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
import matplotlib.pyplot as plt
def plot_ROC(Y,P,P_err=None, cut=-12, title=""):
    hb_prob_Y=np.where(Y<cut, 1, 0)
    hb_prob_P=np.where(P<cut, 1, 0)
    if(P_err is not None):
        hb_prob_P=norm.cdf(cut, loc=P, scale=P_err)
        
        #print(P)
        #print(P_err)
        #print(hb_prob_P)
    
    fpr, tpr, _ = roc_curve(hb_prob_Y, hb_prob_P)
    roc_auc = auc(fpr, tpr)
    
    #plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic{title}")
    plt.legend(loc="lower right")
    plt.gca().set_aspect('equal', 'box')
    #plt.show()
    
def get_fixed_ROC_AUC(Y,P,P_err=None, cut=-12):
    hb_prob_Y=np.where(Y<cut, 1, 0)
    hb_prob_P=np.where(P<cut, 1, 0)
    if(P_err is not None):
        hb_prob_P=norm.cdf(cut, loc=P, scale=P_err)
    
    fpr, tpr, _ = roc_curve(hb_prob_Y, hb_prob_P)
    roc_auc = auc(fpr, tpr)
    return(roc_auc)

#def get_AUC_fixed(Y,P):
#    hb_prob_Y=np.where(Y<cut, 1, 0)


def confinterval(a):
    if(np.isfinite(a).all()):
        mean=np.mean(a)
        sem=st.sem(a)
        if(sem<1e-9): #all values are the same
            return(0)
        return( np.abs(st.t.interval(0.95, len(a)-1, loc=mean, scale=sem)[0]-mean) )
    else:
        return(np.nan)

# print(confinterval([0,0,0,0,0.0]))

def mask_borders(arr, num=1):
    mask = np.zeros(arr.shape, bool)
    for dim in range(arr.ndim):
        mask[tuple(slice(0, num) if idx == dim else slice(None) for idx in range(arr.ndim))] = True
        mask[tuple(slice(-num, None) if idx == dim else slice(None) for idx in range(arr.ndim))] = True
    return mask

def wiener_index(m):
    res = 0
    amat = Chem.GetDistanceMatrix(m)
    num_atoms = m.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            res += amat[i][j]
    return res

def get_feature_score_vector(lig_fmap, xray_fmap):
    #http://rdkit.blogspot.com/2017/11/using-feature-maps.html
    #https://link.springer.com/article/10.1007/s10822-006-9085-8
    vec=np.zeros(xray_fmap.GetNumFeatures())
    for f in range(len(vec)):
        xray_feature=xray_fmap.GetFeature(f)
        vec[f]=np.sum([lig_fmap.GetFeatFeatScore(lig_fmap.GetFeature(f_l), xray_feature) for f_l in range(lig_fmap.GetNumFeatures())])
    return(vec)

def ndmesh(*xi,**kwargs):
    if len(xi) < 2:
        msg = 'meshgrid() takes 2 or more arguments (%d given)' % int(len(xi) > 0)
        raise ValueError(msg)

    args = np.atleast_1d(*xi)
    ndim = len(args)
    copy_ = kwargs.get('copy', True)

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1::]) for i, x in enumerate(args)]

    shape = [x.size for x in output]

    # Return the full N-D matrix (not only the 1-D vector)
    if copy_:
        mult_fact = np.ones(shape, dtype=int)
        return [x * mult_fact for x in output]
    else:
        return np.broadcast_arrays(*output)
