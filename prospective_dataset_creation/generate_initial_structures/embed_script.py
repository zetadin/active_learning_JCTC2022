from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdmolfiles, rdMolAlign, rdmolops, rdchem, PyMol, Crippen, PropertyMol
#from rdkit import DataStructs
from rdkit import RDLogger

#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Geometry import Point3D
from rdkit.Numerics.rdAlignment import GetAlignmentTransform
#from rdkit.Chem.AtomPairs import Pairs
#from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolTransforms
from copy import deepcopy

import numpy as np
import os
import os.path
import pmx
import random
import re
import scipy as sp
import subprocess
import shutil
import sys
import tempfile
import argparse

from scipy.optimize import linear_sum_assignment

import custom_constrained_embed
from custom_constrained_embed import CustomConstrainedEmbed
from datetime import datetime

try:
    import cPickle as pickle
except:
    import pickle

#Bohr2Ang=0.529177249

#folder="/home/ykhalak/Projects/ML_dG/pde2_dG/"
#xray_ligs_folder="/home/ykhalak/private/pde2_CONFIDENTIAL_do_not_backup/aligned_structures/ligands_only/"


################################ HELPER FUNCTIONS ################################

def overlap_measure(molA, molB):
    confA=molA.GetConformer()
    confB=molB.GetConformer()
    posA=[]
    posB=[]
    for i,a in enumerate(molA.GetAtoms()):
        if(a.GetAtomicNum()>1): #not hydrogens
            posA.append(list(confA.GetAtomPosition(i)))

    for i,a in enumerate(molB.GetAtoms()):
        if(a.GetAtomicNum()>1): #not hydrogens
            posB.append(list(confB.GetAtomPosition(i)))
    posA=np.array(posA)
    posB=np.array(posB)
    
    dif=posA[:,np.newaxis,:]-posB[np.newaxis,:,:]
    dist=np.linalg.norm(dif, axis=2)
    A_ind, B_ind = linear_sum_assignment(dist)
    measure = 0
    for i,a in enumerate(A_ind):
        measure+=dist[a, B_ind[i]]
    return(measure) 





def find_mapping(mol, ref, outpath):    
    #remove old output files
    for f in os.listdir(outpath):
        os.remove(os.path.join(outpath, f))
    
    #dump files
    mol_file=outpath+"/mol.pdb"
    ref_file=outpath+"/ref.pdb"
    with open(mol_file,"w") as f:
        f.write(rdmolfiles.MolToPDBBlock(mol))
    with open(ref_file,"w") as f:
        f.write(rdmolfiles.MolToPDBBlock(ref))
        
    #map atoms with pmx
    
    # params
    i1 = ref_file
    i2 = mol_file
    o1 = '{0}/ref_map.dat'.format(outpath)
    o2 = '{0}/mol_map.dat'.format(outpath)
    opdb1 = '{0}/out_pdb1.pdb'.format(outpath)
    opdb2 = '{0}/out_pdb2.pdb'.format(outpath)
    opdbm1 = '{0}/out_pdbm1.pdb'.format(outpath)
    opdbm2 = '{0}/out_pdbm2.pdb'.format(outpath)
    score = '{0}/score.dat'.format(outpath)
    log = '{0}/mapping.log'.format(outpath)

    process = subprocess.Popen(['pmx','atomMapping',
                        '-i1',i1,
                        '-i2',i2,
                        '-o1',o1,
                        '-o2',o2,
                        '-opdb1',opdb1,
                        '-opdb2',opdb2,                                        
                        '-opdbm1',opdbm1,
                        '-opdbm2',opdbm2,
                        '-score',score,
                        '-log',log,
                        '--dMCS', '--d', '0.1',
                        #'--RingsOnly'
                               ],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
    process.wait()
    
    if(not os.path.isfile(o2) ): #mapping failed, use less restrictive match criteria: no distance criterion in MCS
        process = subprocess.Popen(['pmx','atomMapping',
                        '-i1',i1,
                        '-i2',i2,
                        '-o1',o1,
                        '-o2',o2,
                        '-opdb1',opdb1,
                        '-opdb2',opdb2,                                        
                        '-opdbm1',opdbm1,
                        '-opdbm2',opdbm2,
                        '-score',score,
                        '-log',log,
                               ],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        process.wait()
    
    if(not os.path.isfile(o2) ):
        raise RuntimeError('atomMapping failed after a second, less restrictive, attempt.')
    
    #read mapping: indeces of mol ordered as ref
    mol_inds=[]
    ref_inds=[]
    with open(o2,"r") as f:
        for line in f:
            m,r=line.split()
            mol_inds.append(int(m)-1)
            ref_inds.append(int(r)-1)
            
    #the above mapping is in output atom order
    #pmx atomMapping can change the order from the input one though.
            
    with open(score,"r") as f:
        for line in f:
            score_val=float(line.split()[-1])
            break;
            
    return(mol_inds, ref_inds, score_val)






################################ MAIN ################################
def main(fname, debug=False, redo=False):
    xray, scaffold, ref = pickle.load( open( fname, "rb" ) )
    
    print(f"Processing {ref.GetProp('ID')} at {datetime.now()}", flush=True)
    
    #check if previously handled
    if(ref.HasProp('embedded') and ref.GetProp('embedded')=="yes" and not redo):
        print("ready")
        return()
    #messedup overwritten, but already embedded ligands
    elif((ref.HasProp('# xray matched atoms') or ref.HasProp('# ligand matched atoms') or ref.HasProp('xray_to_lig_dist')) and not redo):
        ref.SetProp('embedded', "yes")
        pickle.dump( (xray, scaffold, PropertyMol.PropertyMol(ref)), open( fname, "wb" ) )
        print("ready")
        return()
    elif(ref.HasProp('corrupt') and ref.GetProp('corrupt')=="yes"):
        print("Molecule likely has corrupted rings. Ignoring.")
        return()

    #xray already has hydrogens
    #xray = Chem.AddHs(xray_ligs[xray_id], addCoords=True)
    
    #add Hs to ligand
    ref = Chem.AddHs(ref)
    
    nFullTries=0
    maxFullTries=5
    bestConf_coords=None
    bestDist=None
    
    #n_attempts=100
    n_attempts=10
    
    while (nFullTries<maxFullTries):
        scaffold = None
        
        AllChem.EmbedMolecule(ref)
        if(len(list(ref.GetConformers()))==0):
            ref.SetProp('corrupt', "yes")
            pickle.dump( (xray, scaffold, PropertyMol.PropertyMol(ref)), open( fname, "wb" ) )
            print("Molecule likely has corrupted rings. Ignoring.")
            return()
            #raise(Exception("Could not generate a conformer. Molecule likely has corrupted rings."))

        #do Crippen based alignment first
        ref_crippen = Crippen._GetAtomContribs(ref)
        xray_crippen = Crippen._GetAtomContribs(xray)
        pyO3A = AllChem.GetCrippenO3A(ref, xray, ref_crippen, xray_crippen,options=0) # mol1=probe, mol2=ref
        pyO3A.Align()

        #find atom mappings
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_match, xray_match, score = find_mapping(ref, xray, outpath = temp_dir)
        ref.SetProp('score', str(score))

        #filter out mismatching elements
        ref_atoms=ref.GetAtoms()
        xray_atoms=xray.GetAtoms()
        mishmatches=[]
        for l in range(len(ref_match)):
            rm=ref_match[l]
            xm=xray_match[l]
            #if(ref_atoms[rm].GetAtomicNum()!=xray_atoms[xm].GetAtomicNum()):
            #    mishmatches.append(l)
            #elif(ref_atoms[rm].GetHybridization()!=xray_atoms[xm].GetHybridization()):
            if(ref_atoms[rm].GetHybridization()!=xray_atoms[xm].GetHybridization()):
                mishmatches.append(l)
        if(len(mishmatches)>0):
            xray_match=[xray_match[x] for x in range(len(xray_match)) if not x in mishmatches]
            ref_match=[ref_match[x] for x in range(len(ref_match)) if not x in mishmatches]

        #sanity check
        if(len(xray_match)<4):
            print("WARNING: only {} common atoms found between ligand {} and its and xray".format(
                    len(xray_match), ref.GetProp('ID')        ), flush=True)
#                 raise()
            #don't do the scaffold, it's not worth it, just align, and give up
            ref.SetProp('embedded', "yes")
            bestDist=True # so we don't get error at end
            break;

        #find bonds that belong to the scaffold from among the ones in xray
        scaffold_bonds=[]
        xray_bonds=xray.GetBonds()
        for b_ind,b in enumerate(xray_bonds):
            if(b.GetBeginAtomIdx() in xray_match and b.GetEndAtomIdx() in xray_match):
                scaffold_bonds.append(b_ind)
                
        #remove disconnected atoms
        mishmatches=[]

        for l in range(len(ref_match)):
            rm=ref_match[l]
            xm=xray_match[l]
            connected=False
            for b in xray_bonds:
                if((b.GetBeginAtomIdx()==xm and b.GetEndAtomIdx() in xray_match) or
                (b.GetEndAtomIdx()==xm and b.GetBeginAtomIdx() in xray_match) ):
                    connected=True
                    break;
                
            if(not connected):
                if(debug):
                    print("disconnected atom: ref atom #{}\t xray atom #{}".format(rm, xm))
                mishmatches.append(l)
                
        if(len(mishmatches)>0):
            xray_match=[xray_match[x] for x in range(len(xray_match)) if not x in mishmatches]
            ref_match=[ref_match[x] for x in range(len(ref_match)) if not x in mishmatches]

        #find scaffold from xray and the bonds
        scaffold=Chem.PathToSubmol(xray, scaffold_bonds)

        #check if ref and xray are the same molecule
        if(len(xray_match)==len(ref_match) and len(xray_match)==ref.GetNumAtoms()):
            #then set ref's conformer directly
            ref_conf = ref.GetConformer()
            matches  = xray.GetSubstructMatch(ref)
            for i, match in enumerate(matches):
                ref_conf.SetAtomPosition(i, xray_conf.GetAtomPosition(match))
            bestDist=True # so we don't get error at end
            break;


        #try many constrained conformers and pick the best
        algMap=list(zip(ref_match,xray_match))
        coordMap={}
        xray_conf=xray.GetConformer()
        for i in range(len(xray_match)):
            c=xray_conf.GetAtomPosition(xray_match[i])
            coordMap[ref_match[i]]=c
            if(debug):
                print("lig atom {} @ {}".format(ref_match[i], [c.x, c.y, c.z]))
                
        matches_saved=False; # save xray_match and ref_match only once per try, and only if conformer is better than previous tries
        for attempt in range(n_attempts):
            try:
                CustomConstrainedEmbed(ref, xray, coordMap=coordMap, algMap=algMap, randomseed=-1, debug=debug) #new version using xray+algMap
                #CustomConstrainedEmbed(ref, scaffold, randomseed=-1, coordMap=coordMap, debug=debug) #old version using scaffold (atom order in scaffold does not follow that in xray_match, so final alignment is off)
                #AllChem.ConstrainedEmbed(ref, scaffold, randomseed=-1)
                
                rms = rdMolAlign.AlignMol(ref,xray,atomMap=algMap)
                dist = overlap_measure(ref, xray)

                if(bestConf_coords is None or dist<bestDist):
                    bestConf_coords = ref.GetConformer().GetPositions()
                    bestDist = dist
                    if(not matches_saved):
                        ref.SetProp('# xray matched atoms', str(xray_match))
                        ref.SetProp('# ligand matched atoms', str(ref_match))
                        matches_saved=True
            except (ValueError,RuntimeError) as e:
#                print("failure for xray # {} and ligand # {}.".format(xray_id, ref_id[0]), flush=True)
                if(debug):
                    raise(e)
                continue;

        if(bestDist is None): #failed to find a single viable conformer, retry from the begining
            #print("bestDist is None")
            nFullTries+=1
            continue;
            
        #sucess, stop retrying
        #break;
        
        #retry more initial configurations. Maybe one gives us a better final conf.
        nFullTries+=1

    if(nFullTries>=maxFullTries and bestDist is None):
        raise RuntimeError("Too many embedding failures for ligand {}. Even after regenerating initial structures".format(ref.GetProp('ID')) )
    
    #overwrite conformer coords with the ones from best conformer
    ref_conf = ref.GetConformer()
    for i in range(ref.GetNumAtoms()):
        ref_conf.SetAtomPosition(i, bestConf_coords[i,:])
        
#    #make sure ref is aligned. WHY DO WE NEED THIS? WE ALREADY DO THIS IN CustomConstrainedEmbed !!!!!!
#    algMap=[(j,i) for i,j in enumerate(ref_match)]
#    rms = rdMolAlign.AlignMol(ref,scaffold,atomMap=algMap)
        
    ref.SetProp('embedded', "yes")
    ref.SetProp('xray_to_lig_dist', repr(bestDist))
    pickle.dump( (xray, scaffold, PropertyMol.PropertyMol(ref)), open( fname, "wb" ) )
    
    print(f"Done with {ref.GetProp('ID')} at {datetime.now()}\n", flush=True)



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
    parser = argparse.ArgumentParser(description='Generaes a ligand structure based on a reference xray structure.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', dest='fname', type=str, nargs='+',
                       help='input/output pickle file')
    parser.add_argument('--debug', dest='debug', type=str2bool, nargs='?', const=True, default=False,
                        help="Activate debug mode.")
    parser.add_argument('--redo', dest='redo', type=str2bool, nargs='?', const=True, default=False,
                        help="Recalculate listed ligands, erven if done previously.")
    
    args = parser.parse_args()

    for f in args.fname:
        try:
            main(f, args.debug, args.redo)
        except RuntimeError as e:
            sys.stderr.write(repr(e)+'\n')
            
    exit(0);
