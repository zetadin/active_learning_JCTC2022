import errno
import numpy as np
import os
import gc
from enum import Enum, auto
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdmolfiles, rdMolAlign, rdmolops, rdchem, rdMolDescriptors, ChemicalFeatures
from rdkit.Chem import PeriodicTable, GetPeriodicTable
from rdkit import RDConfig
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.DataStructs import cDataStructs
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
import rdkit.Chem.EState.EState_VSA
import rdkit.Chem.GraphDescriptors
from inspect import getmembers, isfunction, getfullargspec
import glob
import h5py
import hashlib

import oddt
from oddt import toolkit
from oddt import fingerprints
from oddt import interactions

try:
    import cPickle as pickle
except:
    import pickle

from read_cube import read_cube
#from utils import *
from utils import wiener_index, get_feature_score_vector, mask_borders, ndmesh

from rdkit.Chem import rdRGroupDecomposition as rdRGD
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class dataBlocks(Enum):
    #def _generate_next_value_(name, start, count, last_values):
        #print(name, start, count, last_values)
        #raise()
        #return count-1
    
    ESP_full_grid = 0
    ESP_walls = auto()
    ESP_shells = auto()
    ESP_blanked_core = auto()
    ESP_flat_after_limit = auto()
    atom_hot = auto()
    atom_hot_noH = auto()
    MACCS = auto()
    MorganFP = auto()
    rdkitFP = auto()
    minFeatFP = auto()
    shape3D = auto()
    
    Descriptors = auto()
    EState_FP = auto()
    Graph_desc = auto()
    Pharmacophore_feature_map = auto()
    
    #extras
    MOE = auto()
    MQN = auto()
    GETAWAY = auto()
    AUTOCORR2D = auto()
    AUTOCORR3D = auto()
    BCUT2D = auto()
    WHIM = auto()
    RDF = auto()
    USR = auto()
    USRCUT = auto()
    PEOE_VSA = auto()
    SMR_VSA = auto()
    SlogP_VSA = auto()
    MORSE = auto()
    
    # ESP extras
    ESP_on_vdw_surf = auto()
    ESP_core_mask = auto()
    
    #PLEC from oddt
    PLEC = auto()
    PLEC_filtered = auto()
    
    #MD residue-ligand interaction energies (SR Coul. & vdW)
    MDenerg = auto()
    MDenerg_binned = auto()
    MDenerg_longcut = auto()
    MDenerg_longcut_binned = auto()
    
    atom_hot_on_vdw_surf = auto()
    
    #no_core versions
    no_core_atom_hot = auto()
    no_core_MACCS = auto()
    no_core_MorganFP = auto()
    no_core_rdkitFP = auto()
    no_core_minFeatFP = auto()
    no_core_shape3D = auto()
    no_core_Descriptors = auto()
    no_core_EState_FP = auto()
    no_core_Graph_desc = auto()
    no_core_Pharmacophore_feature_map = auto()
    no_core_MOE = auto()
    no_core_MQN = auto()
    no_core_GETAWAY = auto()
    no_core_AUTOCORR2D = auto()
    no_core_AUTOCORR3D = auto()
    no_core_BCUT2D = auto()
    no_core_WHIM = auto()
    no_core_RDF = auto()
    no_core_USR = auto()
    no_core_USRCUT = auto()
    no_core_PEOE_VSA = auto()
    no_core_SMR_VSA = auto()
    no_core_SlogP_VSA = auto()
    no_core_MORSE = auto()
    no_core_PLEC = auto()
    no_core_PLEC_filtered = auto()
    no_core_MDenerg = auto()
    no_core_MDenerg_binned = auto()
    no_core_MDenerg_longcut = auto()
    no_core_MDenerg_longcut_binned = auto()
    no_core_atom_hot_on_vdw_surf = auto()
    
    def __int__(self):
        return self.value
    
def no_core_special_flags():
    specials=[s.value for s in dataBlocks if "no_core_MDenerg" in s.name]
    specials+=[dataBlocks.no_core_atom_hot_on_vdw_surf.value,
               dataBlocks.no_core_PLEC_filtered.value,
               dataBlocks.no_core_GETAWAY.value]
    return(specials)
def no_core_flags():
    return([s.value for s in dataBlocks if "no_core_" in s.name])
def ESP_flags():
    return([s.value for s in dataBlocks if "ESP" in s.name])
def atom_hot_flags():
    return([s.value for s in dataBlocks if "atom_hot" in s.name])
def no_core_atom_hot_flags():
    return([s.value for s in dataBlocks if "no_core_atom_hot" in s.name])

    

class CustomMolModularDataset(Dataset):
    def __init__(self, ligs, no_core_ligs=None,
                 representation_flags=[1]*(len(dataBlocks)-1), molecular_db_file=None,
                 out_folder=os.path.split(os.path.realpath(__file__))[0], datafolder=os.path.split(os.path.realpath(__file__))[0],
                 Temp=300., normalize_x=False, grid_spacing=2, grid_padding=4,
                 X_filter=None, cachefolder=None, verbose=False, use_cache=True, use_hdf5_cache=False, use_combined_cache=True,
                 mdenerg_nbins=10, internal_cache_maxMem_MB=512,
                 morphes_sim_an_folder="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09/morphes_sim_an"): #, mdenerg_ranges=None
        self.representation_flags=representation_flags
        self.active_flags=np.where(self.representation_flags)[0]
        self.out_folder=out_folder
        self.datafolder=datafolder
        self.RT=0.001985875*Temp #kcal/mol
        self.normalize_x=normalize_x
        self.norm_mu=None
        self.norm_width=None
        self.grid_spacing=grid_spacing # A
        self.grid_padding=grid_padding      # A
        self.X_filter=None
        self.internal_filtered_cache=None
        self.verbose=verbose
        self.use_cache=use_cache
        self.use_hdf5_cache=use_hdf5_cache
        self.use_combined_cache=use_combined_cache
        
        self._internal_cache_maxMem=internal_cache_maxMem_MB*1024*1024 # 512 MB by default
        
        self.morphes_sim_an_folder=morphes_sim_an_folder
        
        if(X_filter):
            if(not os.path.exists(X_filter)):
                raise(Exception(f"No such file: {X_filter}"))
            with open(X_filter, 'rb') as f:
                self.X_filter=pickle.load(f)
                #print(self.X_filter)
                #raise()
#         if(ligs is None):
#             if(not molecular_db_file or not os.path.is_file(molecular_db_file)):
#                 raise(Exception("No ligs and no/nonexistent molecular_db_file provided:", molecular_db_file))
#             self.molecular_db_file = molecular_db_file #folder+"/../processed_ligs_em.pickle"
#             with open(self.molecular_db_file, 'rb') as f:
#                 self.ligs = pickle.load(f)
#                 #filter ligands by em completition
#                 #ligs=[l for l in ligs if (l.HasProp('em done') and l.GetProp('em done')=="yes")]
#                 #filter out the ligands with measurements beyond sensitivity limits
#                 ligs=[l for l in ligs if not("<" in l.GetProp("[Q] hPDE2_pIC50") or ">" in l.GetProp("[Q] hPDE2_pIC50"))]
#         else:
        self.ligs=ligs
        self.no_core_ligs=None
        if(any([self.representation_flags[f] for f in no_core_flags()])):
            if(no_core_ligs is not None):
                self.no_core_ligs=no_core_ligs
            else: #build a set of core-less ligands
                core_smiles="c7(C)nc8ccccc8n8c(c6c(Cl)[cH][cH][cH][cH]6)nnc78"
                core=Chem.MolFromSmiles(core_smiles)

                with suppress_stdout_stderr():
                    res,unmatched = rdRGD.RGroupDecompose([core], ligs, asSmiles=True)# print(unmatched)
                if(len(unmatched)>0):
                    raise()

                self.no_core_ligs=[]
                for i,l in enumerate(res):
                    new_l={}
                    for rg in l:
                        if rg=='Core':
                            continue;
                        if l[rg][:3]!='[H]':
                            new_l[rg]=l[rg]
                    if(len(new_l)==1):
                        Rgroup_smi=new_l['R1']
                        smi_params=Chem.rdmolfiles.SmilesParserParams()
                        smi_params.removeHs=False
                        Rgroup=Chem.AddHs(Chem.MolFromSmiles(Rgroup_smi))
                        Rgroup_smi=Chem.MolToSmiles(Rgroup)
                        Rgroup_smi=Rgroup_smi.replace("[*:1]", "")
                        Rgroup_smi=Rgroup_smi.replace("()", "")
                        Rgroup=Chem.MolFromSmiles(Rgroup_smi, smi_params)

                        #get the substructure mapping
                        mapping=list(ligs[i].GetSubstructMatch(Rgroup))

                        # embed the Rgroup
                        with suppress_stdout_stderr():
                            AllChem.EmbedMolecule(Rgroup)

                        # make sure the atoms have the same coordinates
                        RC=Rgroup.GetConformer()
                        lpos=ligs[i].GetConformer().GetPositions()[mapping,:]
                        for j in range(Rgroup.GetNumAtoms()):
                            RC.SetAtomPosition(j, lpos[j,:])

                        # copy properties
                        keys=list(ligs[i].GetPropNames())
                        for k in keys:
                            Rgroup.SetProp(k, ligs[i].GetProp(k))

                        self.no_core_ligs.append(Chem.PropertyMol.PropertyMol(Rgroup))

        if(self.representation_flags[int(dataBlocks.minFeatFP)]):
            fdefName = self.out_folder+'/MinimalFeatures.fdef'
            featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
            self.sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3)
            self.sigFactory.SetBins([(0,2),(2,5),(5,8)])
            self.sigFactory.Init()

#         if(self.representation_flags[int(dataBlocks.atom_hot)] or
#            self.representation_flags[int(dataBlocks.atom_hot_noH)] or
#            self.representation_flags[int(dataBlocks.atom_hot_on_vdw_surf)] or
#            self.representation_flags[int(dataBlocks.ESP_full_grid)] or
#            self.representation_flags[int(dataBlocks.ESP_on_vdw_surf)] or
#            self.representation_flags[int(dataBlocks.ESP_core_mask)] or
#            self.representation_flags[int(dataBlocks.ESP_walls)] or
#            self.representation_flags[int(dataBlocks.ESP_shells)] or
#            self.representation_flags[int(dataBlocks.ESP_blanked_core)] or
#            self.representation_flags[int(dataBlocks.ESP_flat_after_limit)]):
        if(any([self.representation_flags[f] for f in ESP_flags()]) or any([self.representation_flags[f] for f in atom_hot_flags()])):
            self.resolution=2 #A
            self.max_coords=[]
            self.min_coords=[]
            for l in ligs:
                try:
                    conf=l.GetConformer()
                except ValueError as e:
                    print("Could not load conformer for ligand", l.GetProp("ID"))
                    raise(e)
                pos=conf.GetPositions()
                self.min_coords.append(np.min(pos, axis=0))
                self.max_coords.append(np.max(pos, axis=0))
            self.min_coords=np.min(np.array(self.min_coords), axis=0) - self.resolution/2
            self.max_coords=np.max(np.array(self.max_coords), axis=0) + self.resolution/2
            self.gridsize=np.ceil((self.max_coords-self.min_coords)/self.resolution).astype(int)
#             print(self.gridsize)
            
            if(any([self.representation_flags[f] for f in atom_hot_flags()])):
                self.an2ati_map={}
                self.an2ati_map_noH={}
                for lig in ligs:
                    for a in lig.GetAtoms():
                        an=a.GetAtomicNum()
                        if(not an in self.an2ati_map.keys()):
                            self.an2ati_map[an]=len(self.an2ati_map)
                            if(an!=1):
                                self.an2ati_map_noH[an]=len(self.an2ati_map_noH)
    #             print(self.an2ati_map)
    
            # no_core_atom_hot grids
            if(any([self.representation_flags[f] for f in no_core_atom_hot_flags()])):
                self.no_core_max_coords=[]
                self.no_core_min_coords=[]
                for l in self.no_core_ligs:
                    try:
                        conf=l.GetConformer()
                    except ValueError as e:
                        print("Could not load conformer for nio_core ligand", l.GetProp("ID"))
                        raise(e)
                    pos=conf.GetPositions()
                    self.no_core_min_coords.append(np.min(pos, axis=0))
                    self.no_core_max_coords.append(np.max(pos, axis=0))
                self.no_core_min_coords=np.min(np.array(self.no_core_min_coords), axis=0) - self.resolution/2
                self.no_core_max_coords=np.max(np.array(self.no_core_max_coords), axis=0) + self.resolution/2
                self.no_core_gridsize=np.ceil((self.no_core_max_coords-self.no_core_min_coords)/self.resolution).astype(int)

        if(self.representation_flags[int(dataBlocks.shape3D)]):
            x_ = np.linspace(0., self.gridsize[0]*self.resolution, self.gridsize[0], endpoint=False)
            y_ = np.linspace(0., self.gridsize[0]*self.resolution, self.gridsize[1], endpoint=False)
            z_ = np.linspace(0., self.gridsize[0]*self.resolution, self.gridsize[2], endpoint=False)

            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            self.vox_p=np.moveaxis(np.array([x,y,z]), [0], [3])
        
        if(self.representation_flags[int(dataBlocks.Pharmacophore_feature_map)]
                or self.representation_flags[int(dataBlocks.no_core_Pharmacophore_feature_map)]):
            xray_fmf="/home/ykhalak/Projects/ML_dG/pde2_dG/how_do_ligs_fit_in_pocket/adaptive_learning_test_from_morphed_structs/repr_cache/xray_feature_maps.pickle"
            with open(xray_fmf, 'rb') as f:
                self.ref_feature_maps=pickle.load(f)
                
            self.featmap_fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef'))
            self.fmParams = {}
            for k in self.featmap_fdef.GetFeatureFamilies():
                fparams = FeatMaps.FeatMapParams()
                self.fmParams[k] = fparams   
                
        # active site vdw surface
        if(self.representation_flags[int(dataBlocks.ESP_on_vdw_surf)] or self.representation_flags[int(dataBlocks.atom_hot_on_vdw_surf)] ):
            #self.vdw_points=np.loadtxt("/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/4d09_vdw_surface_around_whole_ligs.dat")
            self.vdw_points=np.loadtxt("/home/ykhalak/Projects/ML_dG/pde2_dG/how_do_ligs_fit_in_pocket/adaptive_learning_test_from_morphed_structs/pde2_active_site_vdw_surface.dat")
            
            # make sure we only use points within the grid
            p_mask=[]
            for i in range(3): # check for points within min/max of grid for eacha axis
                p_mask.append(np.logical_and(self.vdw_points[:,i]>self.min_coords[i], self.vdw_points[:,i]<self.max_coords[i]))
            p_mask = np.logical_and(np.logical_and(p_mask[0], p_mask[1]), p_mask[2]) # bundle mask for all axis together
            self.vdw_points=self.vdw_points[np.where(p_mask)] # filter points by mask
            
            # find their grid coordinates
            self.vdw_points_on_grid=np.floor((self.vdw_points-self.min_coords)/self.resolution).astype(int)
            self.vdw_points_on_grid=np.unique(self.vdw_points_on_grid, axis=0)
            
            
        if(self.representation_flags[int(dataBlocks.no_core_atom_hot_on_vdw_surf)] ):
            self.no_core_vdw_points=np.loadtxt("/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/4d09_vdw_surface_around_Rgroups.dat")
            raise()
            
                        # make sure we only use points within the grid
            p_mask=[]
            for i in range(3): # check for points within min/max of grid for eacha axis
                p_mask.append(np.logical_and(self.no_core_vdw_points[:,i]>self.no_core_min_coords[i], self.no_core_vdw_points[:,i]<self.no_core_max_coords[i]))
            p_mask = np.logical_and(np.logical_and(p_mask[0], p_mask[1]), p_mask[2]) # bundle mask for all axis together
            self.no_core_vdw_points=self.no_core_vdw_points[np.where(p_mask)] # filter points by mask
            
            # find their grid coordinates
            self.no_core_vdw_points_on_grid=np.floor((self.no_core_vdw_points-self.no_core_min_coords)/self.resolution).astype(int)
            self.no_core_vdw_points_on_grid=np.unique(self.no_core_vdw_points_on_grid, axis=0)
            
        if(self.representation_flags[int(dataBlocks.PLEC)] or self.representation_flags[int(dataBlocks.no_core_PLEC)]):
            #self.protein=next(toolkit.readfile( 'pdb', '/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/data/setup/prot.pdb'))
            self.protein=next(toolkit.readfile( 'pdb', '/home/ykhalak/Projects/ML_dG/pde2_dG/how_do_ligs_fit_in_pocket/data/setup/prot.pdb'))
            self.protein.protein = True
        if(self.representation_flags[int(dataBlocks.PLEC_filtered)]):
            #self.protein=next(toolkit.readfile( 'pdb', '/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/data/setup/prot.pdb'))
            self.protein=next(toolkit.readfile( 'pdb', '/home/ykhalak/Projects/ML_dG/pde2_dG/how_do_ligs_fit_in_pocket/data/setup/prot.pdb'))
            self.protein.protein = True
            #self.PLEC_contact_filter=np.loadtxt('/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/4d09_PLEC_filter.dat', dtype=int)
            self.PLEC_contact_filter=[ 600,  603,  604,  611,  612,  672,  963,  987,  992,  995, 1011, 1012, 1016, 1072,
                                       1096, 1100, 1104, 1107, 1123, 1168, 1176, 1182, 1203, 1504, 1507, 1508, 1515, 1516,
                                       1520, 1523, 1528, 1531, 1544, 1552, 1800, 1803, 1824, 1828, 1830, 1832, 1848, 1852,
                                       1854, 1856, 1859, 1915, 1936, 1944, 1947, 1955, 1968, 1976, 1979, 1980, 1992, 1996,
                                       2000, 2001, 2002, 2003, 2024, 2120, 2128, 2136, 2200, 2224, 2228, 2232, 2235, 2236,
                                       2251, 2252, 2256, 2257, 2258, 2259, 2288, 2296, 2320, 2520, 2523]
        if(self.representation_flags[int(dataBlocks.no_core_PLEC_filtered)]):
            self.protein=next(toolkit.readfile( 'pdb', '/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/data/setup/prot.pdb'))
            self.protein.protein = True
            self.no_core_PLEC_contact_filter=np.loadtxt('/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09_filtered_subset/4d09_no_core_PLEC_filter.dat', dtype=int)
            raise()


        #binning for MDenerg
        if(self.representation_flags[int(dataBlocks.MDenerg_binned)]):
            self.MDenerg_nbins=mdenerg_nbins
            self.MDenerg_ranges=[]
#            if(mdenerg_ranges is not None):
#                self.MDenerg_ranges=mdenerg_ranges # indexed as [lid_id, min/max]
#            else:
#                raise(Exception("need mdenerg_ranges specified for binning"))
            temp_repr_flags=[0]*len(dataBlocks)
            temp_repr_flags[int(dataBlocks.MDenerg)]=1
            
            temp_DS=CustomMolModularDataset(ligs=self.ligs, no_core_ligs=self.no_core_ligs,
                        representation_flags=temp_repr_flags, molecular_db_file=molecular_db_file,
                        out_folder=out_folder, datafolder=datafolder, Temp=Temp,
                        normalize_x=False, X_filter=None, cachefolder=cachefolder, verbose=False, use_cache=use_cache)
            self.MDenerg_ranges=temp_DS.find_ranges()
            
            del temp_repr_flags,temp_DS
            _=gc.collect()
            
            self.MDenerg_widths=self.MDenerg_ranges[:,1]-self.MDenerg_ranges[:,0]
            self.MDenerg_steps=self.MDenerg_widths/self.MDenerg_nbins
            self.MDenerg_steps[np.abs(self.MDenerg_steps)<1e-7]=1e-7 # deal with widths of 0
            
        if(self.representation_flags[int(dataBlocks.MDenerg_longcut_binned)]):
            self.MDenerg_longcut_nbins=mdenerg_nbins
            self.MDenerg_longcut_ranges=[]
#            if(mdenerg_ranges is not None):
#                self.MDenerg_ranges=mdenerg_ranges # indexed as [lid_id, min/max]
#            else:
#                raise(Exception("need mdenerg_ranges specified for binning"))
            temp_repr_flags=[0]*len(dataBlocks)
            temp_repr_flags[int(dataBlocks.MDenerg_longcut)]=1
            
            temp_DS=CustomMolModularDataset(ligs=self.ligs, no_core_ligs=self.no_core_ligs,
                        representation_flags=temp_repr_flags, molecular_db_file=molecular_db_file,
                        out_folder=out_folder, datafolder=datafolder, Temp=Temp,
                        normalize_x=False, X_filter=None, cachefolder=cachefolder, verbose=False, use_cache=use_cache)
            self.MDenerg_longcut_ranges=temp_DS.find_ranges()
            
            del temp_repr_flags,temp_DS
            _=gc.collect()
            
            self.MDenerg_longcut_widths=self.MDenerg_longcut_ranges[:,1]-self.MDenerg_longcut_ranges[:,0]
            self.MDenerg_longcut_steps=self.MDenerg_longcut_widths/self.MDenerg_longcut_nbins
            self.MDenerg_longcut_steps[np.abs(self.MDenerg_longcut_steps)<1e-7]=1e-7 # deal with widths of 0

        #binning for no_core_MDenerg
        if(self.representation_flags[int(dataBlocks.no_core_MDenerg_binned)]):
            self.no_core_MDenerg_nbins=mdenerg_nbins
            self.no_core_MDenerg_ranges=[]
#            if(mdenerg_ranges is not None):
#                self.MDenerg_ranges=mdenerg_ranges # indexed as [lid_id, min/max]
#            else:
#                raise(Exception("need mdenerg_ranges specified for binning"))
            temp_repr_flags=[0]*len(dataBlocks)
            temp_repr_flags[int(dataBlocks.no_core_MDenerg)]=1
            
            temp_DS=CustomMolModularDataset(ligs=self.ligs, no_core_ligs=self.no_core_ligs,
                        representation_flags=temp_repr_flags, molecular_db_file=molecular_db_file,
                        out_folder=out_folder, datafolder=datafolder, Temp=Temp,
                        normalize_x=False, X_filter=None, cachefolder=cachefolder, verbose=False, use_cache=use_cache)
            self.no_core_MDenerg_ranges=temp_DS.find_ranges()
            
            del temp_repr_flags,temp_DS
            _=gc.collect()
            
            self.no_core_MDenerg_widths=self.no_core_MDenerg_ranges[:,1]-self.no_core_MDenerg_ranges[:,0]
            self.no_core_MDenerg_steps=self.no_core_MDenerg_widths/self.no_core_MDenerg_nbins
            self.no_core_MDenerg_steps[np.abs(self.no_core_MDenerg_steps)<1e-7]=1e-7 # deal with widths of 0
            
        if(self.representation_flags[int(dataBlocks.no_core_MDenerg_longcut_binned)]):
            self.no_core_MDenerg_longcut_nbins=mdenerg_nbins
            self.no_core_MDenerg_longcut_ranges=[]
#            if(mdenerg_ranges is not None):
#                self.MDenerg_ranges=mdenerg_ranges # indexed as [lid_id, min/max]
#            else:
#                raise(Exception("need mdenerg_ranges specified for binning"))
            temp_repr_flags=[0]*len(dataBlocks)
            temp_repr_flags[int(dataBlocks.no_core_MDenerg_longcut)]=1
            
            temp_DS=CustomMolModularDataset(ligs=self.ligs, no_core_ligs=self.no_core_ligs,
                        representation_flags=temp_repr_flags, molecular_db_file=molecular_db_file,
                        out_folder=out_folder, datafolder=datafolder, Temp=Temp,
                        normalize_x=False, X_filter=None, cachefolder=cachefolder, verbose=False, use_cache=use_cache)
            self.no_core_MDenerg_longcut_ranges=temp_DS.find_ranges()
            
            del temp_repr_flags,temp_DS
            _=gc.collect()
            
            self.no_core_MDenerg_longcut_widths=self.no_core_MDenerg_longcut_ranges[:,1]-self.no_core_MDenerg_longcut_ranges[:,0]
            self.no_core_MDenerg_longcut_steps=self.no_core_MDenerg_longcut_widths/self.no_core_MDenerg_longcut_nbins
            self.no_core_MDenerg_longcut_steps[np.abs(self.no_core_MDenerg_longcut_steps)<1e-7]=1e-7 # deal with widths of 0
            
        #prevent nans if no_core GETAWAY
        if(self.representation_flags[int(dataBlocks.no_core_GETAWAY)]):
            self.no_core_GETAWAY_mask = np.ones(len(rdMolDescriptors.CalcGETAWAY(self.no_core_ligs[0])), np.bool)
            #possibles_nans_at = np.array([  1,  52,  53,  92,  93, 119, 123, 137, 143, 159, 205, 209, 235])
            possibles_nans_at = np.array([  1,  52,  53,  92,  93, 119, 123, 205, 209, 261, 263])
            #possibles_nans_at = np.array([  1, 2,  52,  53,  92,  93, 119, 123, 137, 143, 159, 205, 209, 235, 261, 263])
            self.no_core_GETAWAY_mask[possibles_nans_at] = 0
            

        #representation cache path precalc
        self.grid_size_dependent_data_blocks=[]
        for i in range(len(self.representation_flags)):
            if("ESP" in dataBlocks(i).name or "atom_hot" in dataBlocks(i).name):
                self.grid_size_dependent_data_blocks.append(i)
        
        #repr_hash=str(abs(hash(np.array(self.representation_flags, dtype=int).tobytes())))[:5]
        repr_hash=hashlib.md5(np.packbits(np.array(representation_flags, dtype=bool)).tobytes()).hexdigest()
        if(cachefolder is None):
            self.cachefolder=f"{self.datafolder}/combined_modular_repr_cache/{repr_hash}"        
        else:
            self.cachefolder=cachefolder
        if (self.use_combined_cache and not os.path.exists(self.cachefolder)): #make sure the folder exists
            try:
                os.makedirs(self.cachefolder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                    
        # hdf5 caches
        if(self.use_hdf5_cache):
            if not os.path.exists(self.datafolder+"/modular_repr_cache_hdf5"): #make sure the folder exists
                os.makedirs(self.datafolder+"/modular_repr_cache_hdf5")
            self.hdf5_repr_cache_files=[]
            for i in range(len(self.representation_flags)):
                if(self.representation_flags[i]):
                    fn = self.datafolder+"/modular_repr_cache_hdf5/"+dataBlocks(i).name+".hdf5"
                    self.hdf5_repr_cache_files.append( h5py.File(fn,'a'))
                else:
                    self.hdf5_repr_cache_files.append(None)
                    
    def __del__(self):
        # close hdf5 caches files
        if(self.use_hdf5_cache):
            for i in range(len(self.representation_flags)):
                if(self.representation_flags[i]):
                    self.hdf5_repr_cache_files[i].close()
                    
    def find_ranges(self):
        allX=np.array([entry[0] for entry in self])
        allrange=np.zeros((allX.shape[1],2))
        # allX axis 0 loops over ligands
        allrange[:,0]=np.min(allX, axis=0)
        allrange[:,1]=np.max(allX, axis=0)
        return(allrange)

    def find_normalization_factors(self):
        # read the normalization cache if it was previusly saved
        filt_spec="_no_X_filter"
        fn_no_filt=f"{self.cachefolder}/normalization_factors_{filt_spec}.dat"
        if(self.X_filter is not None):
            filt_hash=hashlib.md5(np.packbits(np.array(self.X_filter, dtype=bool)).tobytes()).hexdigest()
            filt_spec="_fiter_hash_"+filt_hash
        fn=f"{self.cachefolder}/normalization_factors_{filt_spec}.dat"
        if(os.path.exists(fn)):
            temp=np.loadtxt(fn)
            temp=temp.astype(np.float32) # defaults to float64, which translates to torch's double and is incompatible with linear layers
            self.norm_mu=temp[0,:]
            self.norm_width=temp[1,:]
            if(self.verbose):
                print(f"Reading normalization factors for a {norm_mu.shape} dataset")
        elif(os.path.exists(fn_no_filt) and self.X_filter is not None):
            temp=np.loadtxt(fn_no_filt)
            temp=temp.astype(np.float32) # defaults to float64, which translates to torch's double and is incompatible with linear layers
            self.norm_mu=temp[0,self.X_filter]
            self.norm_width=temp[1,self.X_filter]
            if(self.verbose):
                print(f"Reading normalization factors for a {norm_mu.shape} dataset")
        else:
            self.normalize_x=False
            allX=np.array([entry[0] for entry in self])
            self.norm_mu=np.mean(allX, axis=0)
            self.norm_width=np.std(allX, axis=0)
            self.norm_width[self.norm_width<1e-7]=1.0 # if standard deviation is 0, don't scale
            self.normalize_x=True
            
            # save normalization factors
            if not os.path.exists(self.cachefolder): #make sure the folder exists
                os.makedirs(self.cachefolder, exist_ok=True)
            np.savetxt(fn, np.vstack((self.norm_mu, self.norm_width)))
            
            if(self.verbose):
                print(f"Generating normalization factors for a {allX.shape} dataset")
        self.build_internal_filtered_cache()
        _=gc.collect()


    def copy_normalization_factors(self, other):
        if(not np.array_equal(self.X_filter,other.X_filter)):
            raise(Exception("Mismatching X_filters!"))
        self.norm_mu=other.norm_mu
        self.norm_width=other.norm_width
        
    def build_internal_filtered_cache(self):
        if(self.norm_mu is None and self.normalize_x):
            raise(Exception("call build_internal_filtered_cache() only after normalization!"))
        neededMem=len(self)*(self[0][0].shape[0]+self[0][1].shape[0])*self[0][1].itemsize
        if(neededMem>self._internal_cache_maxMem):
            print(f"Building the internal_filtered_cache needs {neededMem/1024/1024} MB, more than the {self._internal_cache_maxMem/1024/1024} MB limit. SKIPPING and will read samples from HDD each time instead.")
            return()
        #print("norm_mu:", self.norm_mu.shape)
        #print("norm_width:", self.norm_width.shape)
        #print("repr width:", self.__getitem__(0)[0].shape)
        #allX=np.array([entry[0] for entry in self])
        #allY=np.array([entry[1] for entry in self])
        allX=[]
        allY=[]
        for entry in self: # loop over self only once
            allX.append(entry[0])
            allY.append(entry[1])
        allX=np.array(allX)
        allY=np.array(allY)
        self.internal_filtered_cache=(allX, allY)
        if(self.verbose):
            print(f"saving an internal filtered & normalized cache of shape ({self.internal_filtered_cache[0].shape},{self.internal_filtered_cache[1].shape})")


    def normalize_input(self,x):
        if(self.norm_mu is None):
            self.find_normalization_factors()
        return((x-self.norm_mu)/self.norm_width)



    #def copy_maps(self, other):
        #if(self.representation != other.representation):
            #raise(Exception("Mismatching representations!"))
        #if(self.representation == dataRep.atom_hot or self.representation == dataRep.atom_hot_noH):
            #self.an2ati_map=other.an2ati_map
            #self.an2ati_map_noH=other.an2ati_map_noH
        #else:
            #raise(Exception("represenation is", self.representation.name))


    def __len__(self):
        return len(self.ligs)

    def __getitem__(self, idx):
        lig = self.ligs[idx]

        if(self.internal_filtered_cache is None):
            lig_ID = lig.GetProp("ID")
            #check combined repr cache
            cache_fn = self.cachefolder+'/'+lig_ID+'.pickle'
            if(self.use_combined_cache and os.path.isfile(cache_fn)):
                with open(cache_fn, 'rb') as f:
                    X, Y = pickle.load(f)
            else:
                X = self.transform(idx).astype(np.float32)
                Y = np.array([float(lig.GetProp('dG')) if lig.HasProp('dG') else np.nan]) # kcal/mol
                #save cache
                if(self.use_cache and self.use_combined_cache):
                    with open(cache_fn, 'wb') as f:
                        pickle.dump((X, Y), f)
            #if(self.X_filter):
            if(not self.X_filter is None):
                X=X[self.X_filter]
            if(self.normalize_x):
                #print(f"{lig_ID} width: {X.shape}")
                X=self.normalize_input(X)
                
        else:
            X=self.internal_filtered_cache[0][idx]
            Y=self.internal_filtered_cache[1][idx]
        
        return X, Y
    
        #weight=1.0
        #if(lig.HasProp("ML_weight")):
            #weight=float(lig.GetProp("ML_weight"))

        #return X, Y, weight
        
    def generate_DataBlock(self, lig, blockID):
        blockID=dataBlocks(blockID)
        
        if(blockID==dataBlocks.ESP_full_grid):
            key=lig.GetProp('ID')
            fn=self.out_folder+"/../ESP_from_morphes_sim_an/lig_{}/lig_{}_pad_{}_a_{}.cub".format(
                key,key, self.grid_padding, self.grid_spacing)
            grid=read_cube(fn)["data"]
            return(grid.flatten())
            
        elif(blockID==dataBlocks.ESP_walls):
            key=lig.GetProp('ID')
            fn=self.out_folder+"/../ESP_from_morphes_sim_an/lig_{}/lig_{}_pad_{}_a_{}.cub".format(
                key,key, self.grid_padding, self.grid_spacing)
            grid=read_cube(fn)["data"]

            left=grid
            walls_sum_pos=[]
            walls_sum_neg=[]
            for j in range(int(grid.shape[0]/2)):
                walls=[left[0,:,:], left[-1,:,:],left[:,0,:], left[:,-1,:],left[:,:,0], left[:,:,-1]]
                for w in walls:
                    w=w.flatten()
                    walls_sum_pos.append(w[w>0].sum())
                    walls_sum_neg.append(w[w<=0].sum())
                mask=mask_borders(left, num=1)
                left=left[np.logical_not(mask)].reshape((grid.shape[0]-(j+1)*2, grid.shape[0]-(j+1)*2, grid.shape[0]-(j+1)*2))
            walls_rep=np.array(walls_sum_pos+walls_sum_neg)
            return(walls_rep)
        
        elif(blockID==dataBlocks.ESP_shells):
            key=lig.GetProp('ID')
            fn=self.out_folder+"/../ESP_from_morphes_sim_an/lig_{}/lig_{}_pad_{}_a_{}.cub".format(
                key,key, self.grid_padding, self.grid_spacing)
            grid=read_cube(fn)["data"]
                    #walls and shells representations
            left=grid
            sum_pos=[]
            sum_neg=[]
            for j in range(int(grid.shape[0]/2)):

                mask=mask_borders(left, num=1)
                border=left[mask]
                left=left[np.logical_not(mask)].reshape((grid.shape[0]-(j+1)*2, grid.shape[0]-(j+1)*2, grid.shape[0]-(j+1)*2))
                pos=border[border>0.0]
                neg=border[border<=0.0]
                sum_pos.append(pos.sum())
                sum_neg.append(neg.sum())
            shells_rep=np.array(sum_pos+sum_neg)
            return(shells_rep)
            
        elif(blockID==dataBlocks.ESP_blanked_core):
            key=lig.GetProp('ID')
            fn=self.out_folder+"/../ESP_from_morphes_sim_an/lig_{}/lig_{}_pad_{}_a_{}.cub".format(
                key,key, self.grid_padding, self.grid_spacing)
            cube=read_cube(fn)
            grid=cube["data"]
            grid_axes=[]
            for i in range(3):
                grid_axes.append(cube['uvecs'][i][i]*np.arange(grid.shape[i]))
            a_pos=np.array([a[1] for a in cube['atoms']])
            a_VdW=np.array([rdchem.PeriodicTable.GetRvdw(GetPeriodicTable(), int(a[0])) for a in cube['atoms']])
            a_pos=a_pos.reshape(1,1,1,-1,3)
            a_VdW=a_VdW.reshape(1,1,1,-1)
                
            p_grid=cube['origin'][:3]+np.vstack((ndmesh(*grid_axes))).reshape(3,-1).T
            p_grid=p_grid.reshape(grid.shape[0],grid.shape[1],grid.shape[2],1,3)
            
            d_sq=np.sum((p_grid-a_pos)**2, axis=-1)
            inside_mask=np.any(d_sq<=a_VdW**2, axis=-1)
            grid[inside_mask]=0.0
            return(grid.flatten())
        
        elif(blockID==dataBlocks.ESP_flat_after_limit):
            limit=0.140387
            key=lig.GetProp('ID')
            fn=self.out_folder+"/../ESP_from_morphes_sim_an/lig_{}/lig_{}_pad_{}_a_{}.cub".format(
                key,key, self.grid_padding, self.grid_spacing)
            cube=read_cube(fn)
            grid=cube["data"]
            grid[grid>limit]=limit
            return(grid.flatten())
        
        elif(blockID==dataBlocks.atom_hot):
            atom_hot=np.zeros((self.gridsize[0], self.gridsize[1], self.gridsize[2], len(self.an2ati_map)), dtype=np.uint8)
            conf=lig.GetConformer()
            pos=conf.GetPositions()
            for j,a in enumerate(lig.GetAtoms()):
                an=a.GetAtomicNum()
                p=pos[j]-self.min_coords
                voxel=np.floor((p/self.resolution)).astype(int)
                #if(np.any(voxel>=self.gridsize)):
                    #print(pos, voxel, self.gridsize, flush=True)
                atom_hot[voxel[0],voxel[1],voxel[2], self.an2ati_map[an]]+=1
            return(atom_hot.flatten())
        
        elif(blockID==dataBlocks.atom_hot_noH):
            atom_hot_noH=np.zeros((self.gridsize[0], self.gridsize[1], self.gridsize[2], len(self.an2ati_map_noH)), dtype=np.uint8)
            conf=lig.GetConformer()
            pos=conf.GetPositions()
            for j,a in enumerate(lig.GetAtoms()):
                an=a.GetAtomicNum()
                if(an>1):
                    p=pos[j]-self.min_coords
                    voxel=np.floor((p/self.resolution)).astype(int)
                    #if(np.any(voxel>=self.gridsize)):
                        #print(pos, voxel, self.gridsize, flush=True)
                    atom_hot_noH[voxel[0],voxel[1],voxel[2], self.an2ati_map_noH[an]]+=1
            return(atom_hot_noH.flatten())
        
        elif(blockID==dataBlocks.atom_hot_on_vdw_surf):
            atom_hot=np.zeros((self.gridsize[0], self.gridsize[1], self.gridsize[2], len(self.an2ati_map)), dtype=np.uint8)
            conf=lig.GetConformer()
            pos=conf.GetPositions()
            for j,a in enumerate(lig.GetAtoms()):
                an=a.GetAtomicNum()
                p=pos[j]-self.min_coords
                voxel=np.floor((p/self.resolution)).astype(int)
                #if(np.any(voxel>=self.gridsize)):
                    #print(pos, voxel, self.gridsize, flush=True)
                atom_hot[voxel[0],voxel[1],voxel[2], self.an2ati_map[an]]+=1

            #limit to vdw surface
            atom_hot=np.array([atom_hot[c[0],c[1],c[2],:] for c in self.vdw_points_on_grid])
                
            return(atom_hot.flatten())
            
        
        elif(blockID==dataBlocks.MACCS):
            Chem.GetSymmSSSR(lig)
            MACCS_txt=cDataStructs.BitVectToText(rdMolDescriptors.GetMACCSKeysFingerprint(lig))
            MACCS_arr=np.zeros(len(MACCS_txt), dtype=np.uint8)
            for j in range(len(MACCS_txt)):
                if(MACCS_txt[j]=="1"):
                    MACCS_arr[j]=1;
            return(MACCS_arr)
        
        elif(blockID==dataBlocks.MorganFP):
            Chem.GetSymmSSSR(lig)
            Morgan_txt=cDataStructs.BitVectToText(rdMolDescriptors.GetMorganFingerprintAsBitVect(lig, 2))
            Morgan_arr=np.zeros(len(Morgan_txt), dtype=np.uint8)
            for j in range(len(Morgan_txt)):
                if(Morgan_txt[j]=="1"):
                    Morgan_arr[j]=1;
            return(Morgan_arr)
        
        elif(blockID==dataBlocks.rdkitFP):
            Chem.GetSymmSSSR(lig)
            rdkitFingerprint_txt=cDataStructs.BitVectToText(Chem.rdmolops.RDKFingerprint(lig))
            rdkitFingerprint_arr=np.zeros(len(rdkitFingerprint_txt), dtype=np.uint8)
            for j in range(len(rdkitFingerprint_txt)):
                if(rdkitFingerprint_txt[j]=="1"):
                    rdkitFingerprint_arr[j]=1;
            return(rdkitFingerprint_arr)
        
        elif(blockID==dataBlocks.minFeatFP):
            Chem.GetSymmSSSR(lig)
            minFeatFingerprint_txt=cDataStructs.BitVectToText(Generate.Gen2DFingerprint(lig, self.sigFactory))
            minFeatFingerprint_arr=np.zeros(len(minFeatFingerprint_txt), dtype=np.uint8)
            for j in range(len(minFeatFingerprint_txt)):
                if(minFeatFingerprint_txt[j]=="1"):
                    minFeatFingerprint_arr[j]=1;
            return(minFeatFingerprint_arr)
    
        elif(blockID==dataBlocks.shape3D):
            shape=np.zeros((self.gridsize[0], self.gridsize[1], self.gridsize[2]), dtype=np.uint8)
            conf=lig.GetConformer()
            pos=conf.GetPositions()
            for j,a in enumerate(lig.GetAtoms()):
                an=a.GetAtomicNum()
                cut2=PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), an)**2
                p=pos[j]-self.min_coords
                dist_mat=p[np.newaxis,np.newaxis,np.newaxis,:]-self.vox_p
                dist_mat=np.sum(dist_mat**2, axis=3)
                shape[dist_mat<cut2]=1
            return(shape.flatten())
    
        elif(blockID==dataBlocks.Descriptors):
            nms=[x[0] for x in Descriptors._descList]
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
            des = np.array(calc.CalcDescriptors(lig))
            return(des)
        
        elif(blockID==dataBlocks.EState_FP):
            ES=Fingerprinter.FingerprintMol(lig)
            funcs=getmembers(rdkit.Chem.GraphDescriptors, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            ES_VSA=np.array([f(lig) for f in funcs])
            ES_FP=np.concatenate((ES[0],ES[1],ES_VSA))
            return(ES_FP)
    
        elif(blockID==dataBlocks.Graph_desc):
            funcs=getmembers(rdkit.Chem.GraphDescriptors, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            funcs+=[wiener_index]
            graph_desc=np.array([f(lig) for f in funcs])
            return(graph_desc)
    
        elif(blockID==dataBlocks.Pharmacophore_feature_map):
            rawFeats = self.featmap_fdef.GetFeaturesForMol(lig)
            lig_fm = FeatMaps.FeatMap(feats = rawFeats, weights=[1]*len(rawFeats), params=self.fmParams)
            # build a vector for each xray featuremap
            vecs=[get_feature_score_vector(lig_fm, self.ref_feature_maps[j]) for j in range(len(self.ref_feature_maps))]
            feat_map_FP=np.concatenate(tuple(vecs), axis=0)
            return(feat_map_FP)
        
        #extras
        elif(blockID==dataBlocks.MOE):
            funcs=getmembers(rdkit.Chem.MolSurf, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            MOE=np.array([f(lig) for f in funcs])
            return(MOE)
        elif(blockID==dataBlocks.MQN):
            return(np.array(rdMolDescriptors.MQNs_(lig) ))
        elif(blockID==dataBlocks.GETAWAY):
            return(np.array(rdMolDescriptors.CalcGETAWAY(lig) ))
        elif(blockID==dataBlocks.AUTOCORR2D):
            return(np.array(rdMolDescriptors.CalcAUTOCORR2D(lig) ))
        elif(blockID==dataBlocks.AUTOCORR3D):
            return(np.array(rdMolDescriptors.CalcAUTOCORR3D(lig) ))
        elif(blockID==dataBlocks.BCUT2D):
            return(np.array(rdMolDescriptors.BCUT2D(lig) ))
        elif(blockID==dataBlocks.WHIM):
            return(np.array(rdMolDescriptors.CalcWHIM(lig) ))
        elif(blockID==dataBlocks.RDF):
            return(np.array(rdMolDescriptors.CalcRDF(lig) ))
        elif(blockID==dataBlocks.USR):
            return(np.array(rdMolDescriptors.GetUSR(lig) ))
        elif(blockID==dataBlocks.USRCUT):
            return(np.array(rdMolDescriptors.GetUSRCAT(lig) ))
        elif(blockID==dataBlocks.PEOE_VSA):
            return(np.array(rdMolDescriptors.PEOE_VSA_(lig) ))
        elif(blockID==dataBlocks.SMR_VSA):
            return(np.array(rdMolDescriptors.SMR_VSA_(lig) ))
        elif(blockID==dataBlocks.SlogP_VSA):
            return(np.array(rdMolDescriptors.SlogP_VSA_(lig) ))
        elif(blockID==dataBlocks.MORSE):
            return(np.array(rdMolDescriptors.CalcMORSE(lig) ))
        
        # ESP extras
        elif(blockID==dataBlocks.ESP_on_vdw_surf):
            key=lig.GetProp('ID')
            fn=self.out_folder+"/../ESP_from_morphes_sim_an/lig_{}/lig_{}_pad_{}_a_{}.cub".format(
                key,key, self.grid_padding, self.grid_spacing)
            cube=read_cube(fn)
            grid=cube["data"]
            grid_axes=[]
            for i in range(3):
                grid_axes.append(cube['uvecs'][i][i]*np.arange(grid.shape[i]))
                
            p_grid=cube['origin'][:3]+np.vstack((ndmesh(*grid_axes))).reshape(3,-1).T
            p_grid=p_grid.reshape(grid.shape[0],grid.shape[1],grid.shape[2],1,3)
            
            grid_step=p_grid[1,1,1,0]-p_grid[0,0,0,0]
            grid_start=p_grid[0,0,0,0]
            
            ESP_values=np.array([grid[c[0],c[1],c[2]] for c in self.vdw_points_on_grid])
            return(ESP_values)
        
        elif(blockID==dataBlocks.ESP_core_mask):
            key=lig.GetProp('ID')
            fn=self.out_folder+"/../ESP_from_morphes_sim_an/lig_{}/lig_{}_pad_{}_a_{}.cub".format(
                key,key, self.grid_padding, self.grid_spacing)
            cube=read_cube(fn)
            grid=cube["data"]
            grid_axes=[]
            for i in range(3):
                grid_axes.append(cube['uvecs'][i][i]*np.arange(grid.shape[i]))
            a_pos=np.array([a[1] for a in cube['atoms']])
            a_VdW=np.array([rdchem.PeriodicTable.GetRvdw(GetPeriodicTable(), int(a[0])) for a in cube['atoms']])
            a_pos=a_pos.reshape(1,1,1,-1,3)
            a_VdW=a_VdW.reshape(1,1,1,-1)
                
            p_grid=cube['origin'][:3]+np.vstack((ndmesh(*grid_axes))).reshape(3,-1).T
            p_grid=p_grid.reshape(grid.shape[0],grid.shape[1],grid.shape[2],1,3)
            
            d_sq=np.sum((p_grid-a_pos)**2, axis=-1)
            inside_mask=np.any(d_sq<=a_VdW**2, axis=-1)
            return(inside_mask.flatten())
        
        elif(blockID==dataBlocks.PLEC):
            #key=lig.GetProp('ID')
            #fn=f"{self.morphes_sim_an_folder}/{key}/morph_fit.pdb"
            #ligand=next(toolkit.readfile( 'pdb', fn))
            ligand=toolkit.readstring('pdb', Chem.rdmolfiles.MolToPDBBlock(lig))
            IFP = fingerprints.InteractionFingerprint( ligand, self.protein)
            return(IFP)
        
        elif(blockID==dataBlocks.PLEC_filtered):
            #key=lig.GetProp('ID')
            #fn=f"{self.morphes_sim_an_folder}/{key}/morph_fit.pdb"
            #ligand=next(toolkit.readfile( 'pdb', fn))
            ligand=toolkit.readstring('pdb', Chem.rdmolfiles.MolToPDBBlock(lig))
            IFP = fingerprints.InteractionFingerprint( ligand, self.protein)
            return(IFP[self.PLEC_contact_filter])
            
        elif(blockID==dataBlocks.no_core_PLEC_filtered):
            #key=lig.GetProp('ID')
            #fn=f"{self.morphes_sim_an_folder}/{key}/morph_fit.pdb"
            #ligand=next(toolkit.readfile( 'pdb', fn))
            ligand=toolkit.readstring('pdb', Chem.rdmolfiles.MolToPDBBlock(lig))
            IFP = fingerprints.InteractionFingerprint( ligand, self.protein)
            return(IFP[self.no_core_PLEC_contact_filter])
        
        elif(blockID==dataBlocks.MDenerg):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_prospectivelike_?.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_prospectivelike_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_prospectivelike_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            return(vals)
            
        elif(blockID==dataBlocks.MDenerg_binned):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_prospectivelike_?.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_prospectivelike_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_prospectivelike_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            vals-=self.MDenerg_ranges[:,0]
            vals/=self.MDenerg_steps
            return(vals)
        
        elif(blockID==dataBlocks.MDenerg_longcut):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_long_cut_?.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_long_cut_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_long_cut_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            return(vals)
            
        elif(blockID==dataBlocks.MDenerg_longcut_binned):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_long_cut_*.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_long_cut_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_long_cut_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            vals-=self.MDenerg_longcut_ranges[:,0]
            vals/=self.MDenerg_longcut_steps
            return(vals)
        
        
        #no_core versions   
        elif(blockID==dataBlocks.no_core_MDenerg):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_?.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_nocore_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            return(vals)
            
        elif(blockID==dataBlocks.no_core_MDenerg_binned):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_?.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_nocore_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            vals-=self.no_core_MDenerg_ranges[:,0]
            vals/=self.no_core_MDenerg_steps
            return(vals)
        
        elif(blockID==dataBlocks.no_core_MDenerg_longcut):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_long_cut_?.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_long_cut_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_nocore_long_cut_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            return(vals)
            
        elif(blockID==dataBlocks.no_core_MDenerg_longcut_binned):
            key=lig.GetProp('ID')
            fns=glob.glob(f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_long_cut_*.xvg")
            vals=[]
            for i in range(len(fns)):
                fn=f"{self.morphes_sim_an_folder}/{key}/energy_block_nocore_long_cut_{i+1}.xvg"
                if(not os.path.exists(fn)):
                    raise(Exception(f"ligand {key}: missing energy_block_nocore_long_cut_{i+1}.xvg"))
                vals.append(np.loadtxt(fn)[1:])
            vals=np.hstack(vals)
            vals-=self.no_core_MDenerg_longcut_ranges[:,0]
            vals/=self.no_core_MDenerg_longcut_steps
            return(vals)
            
        elif(blockID==dataBlocks.no_core_atom_hot_on_vdw_surf):
            atom_hot=np.zeros((self.gridsize[0], self.gridsize[1], self.gridsize[2], len(self.an2ati_map)), dtype=np.uint8)
            conf=lig.GetConformer()
            pos=conf.GetPositions()
            for j,a in enumerate(lig.GetAtoms()):
                an=a.GetAtomicNum()
                p=pos[j]-self.no_core_min_coords
                voxel=np.floor((p/self.resolution)).astype(int)
                #if(np.any(voxel>=self.no_core_gridsize)):
                    #print(pos, voxel, self.no_core_gridsize, flush=True)
                atom_hot[voxel[0],voxel[1],voxel[2], self.an2ati_map[an]]+=1

            #limit to vdw surface
            atom_hot=np.array([atom_hot[c[0],c[1],c[2],:] for c in self.no_core_vdw_points_on_grid])
                
            return(atom_hot.flatten())
            
        elif(blockID==dataBlocks.no_core_GETAWAY):
            ret=np.array(rdMolDescriptors.CalcGETAWAY(lig) )
            ret=ret[self.no_core_GETAWAY_mask]
            return(ret)
        

    
        else:
            raise(Exception(f"Unsupported dataBlock requested: {blockID}"))
        
        
            

    def transform(self, lig_idx):
        vecs=[]
        #for i in range(len(self.representation_flags)):
        #    if(self.representation_flags[i]):
                
        for i in self.active_flags:
            #where are the block chaches?
            cache_folder=self.datafolder+"/modular_repr_cache/"+dataBlocks(i).name+"/"
            if(i in self.grid_size_dependent_data_blocks):
                cache_folder=self.datafolder+"/modular_repr_cache/{}_p_{}_a_{}/".format(
                    dataBlocks(i).name, self.grid_padding, self.grid_spacing)
                
            if not os.path.exists(cache_folder): #make sure the folder exists
                try:
                    os.makedirs(cache_folder)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    
            #if block is cached, read it
            lig_ID = self.ligs[lig_idx].GetProp("ID")
            cache_fn = cache_folder+'/'+lig_ID+'.pickle'
            hdf5_tn=f"/{dataBlocks(i).name}/{lig_ID}"
            if(self.use_hdf5_cache and hdf5_tn in self.hdf5_repr_cache_files[i]): #try hdf5 cache first
                X_block_rep=self.hdf5_repr_cache_files[i][hdf5_tn][:]
            elif(os.path.isfile(cache_fn)): #try pickle cache second
                with open(cache_fn, 'rb') as f:
                    X_block_rep = pickle.load(f)
                if(self.use_hdf5_cache): # also make the hdf5 cache from pickles
                    self.hdf5_repr_cache_files[i].create_dataset(hdf5_tn, data=X_block_rep, dtype='f')
            else: #generate a block and cache it otherwize
                if(i in no_core_flags() and i not in no_core_special_flags()):
                    # use the same repr code as with core but change to a core-less ligand
                    j = dataBlocks[dataBlocks(i).name[8:]].value
                    #print(f"mapping {dataBlocks(i).name} ({i}) to {dataBlocks(j).name} ({j})")
                    X_block_rep = self.generate_DataBlock(self.no_core_ligs[lig_idx], j)
                elif(i in no_core_special_flags()):
                    #special implementation for MDenerg with no_core
                    #because it needs differently named .xvg files
                    X_block_rep = self.generate_DataBlock(self.no_core_ligs[lig_idx], i)
                else:
                    X_block_rep = self.generate_DataBlock(self.ligs[lig_idx], i)
                    
                if(self.use_cache):
                    with open(cache_fn, 'wb') as f:
                        pickle.dump(X_block_rep, f)
                if(self.use_hdf5_cache):
                    self.hdf5_repr_cache_files[i].create_dataset(hdf5_tn, data=X_block_rep, dtype='f')
            
            #add to overall representation
            vecs.append(X_block_rep)
        return(np.concatenate(tuple(vecs), axis=0))




