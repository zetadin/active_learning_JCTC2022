import numpy as np
import os
import gc
import sys
import importlib

try:
    import cPickle as pickle
except:
    import pickle

#(re)load parent class when this one is (re)loaded
if 'custom_dataset_modular_with_binning' in sys.modules:
    importlib.reload(sys.modules['custom_dataset_modular_with_binning'])
else:
    import custom_dataset_modular_with_binning
from custom_dataset_modular_with_binning import *

class CustomMolModularDataset_w_uncert(CustomMolModularDataset):
    def __init__(self, *args, **kwargs):
        super(CustomMolModularDataset_w_uncert, self).__init__(*args, **kwargs)
        
    def build_internal_filtered_cache(self):
        if(self.norm_mu is None and self.normalize_x):
            raise(Exception("call build_internal_filtered_cache() only after normalization!"))
        neededMem=len(self)*(self[0][0].shape[0]+self[0][1].shape[0])*self[0][1].itemsize
        if(neededMem>self._internal_cache_maxMem):
            print(f"Building the internal_filtered_cache needs {neededMem/1024/1024} MB, more than the {self._internal_cache_maxMem/1024/1024} MB limit. SKIPPING and will read samples from HDD each time instead.")
            return()
        #allX=np.array([entry[0] for entry in self])
        #allY=np.array([entry[1] for entry in self])
        # don't call __getitem__ twice for each ligand!
        allX=[]
        allY=[]
        allY_err=[]
        for idx in range(len(self)):
            X,Y = super(CustomMolModularDataset_w_uncert, self).__getitem__(idx)
            allX.append(X)
            allY.append(Y)
            allY_err.append(float(self.ligs[idx].GetProp("dG_err")))
        allX=np.array(allX)
        allY=np.array(allY)
        allY_err=np.array(allY_err)
        self.internal_filtered_cache=(allX, allY, allY_err)
        if(self.verbose):
            print(f"saving an internal filtered & normalized cache of shape ({self.internal_filtered_cache[0].shape},{self.internal_filtered_cache[1].shape},{self.internal_filtered_cache[2].shape})")

    def __getitem__(self, idx):
        
        if(self.internal_filtered_cache is None):
            X,Y_mean = super(CustomMolModularDataset_w_uncert, self).__getitem__(idx)
            Y_err = float(self.ligs[idx].GetProp("dG_err"))
        else:
            X = self.internal_filtered_cache[0][idx]
            Y_mean = self.internal_filtered_cache[1][idx]
            Y_err = self.internal_filtered_cache[2][idx]
        
        Y = np.random.normal(Y_mean, Y_err)
        return X, Y




