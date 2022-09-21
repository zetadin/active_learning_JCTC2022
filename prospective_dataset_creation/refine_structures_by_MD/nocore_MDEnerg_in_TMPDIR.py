from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdmolfiles, rdMolAlign, rdmolops, rdchem, rdMolDescriptors, ChemicalFeatures, PeriodicTable, PropertyMol
from rdkit import DataStructs
from rdkit import RDLogger

from rdkit.Chem.Draw import IPythonConsole
from rdkit.Geometry import Point3D
from rdkit.Numerics.rdAlignment import GetAlignmentTransform
from rdkit.DataStructs import cDataStructs

from tqdm import tqdm, trange
from copy import deepcopy

import argparse
import numpy as np
import shutil
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime

import pmx
from pmx import ndx

from rdkit.Chem import rdRGroupDecomposition as rdRGD
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)




try:
    import cPickle as pickle
except:
    import pickle 

pocket_fit_folder="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09/"
tmpdir=os.environ['TMPDIR']


############################################################
# step by step functions
############################################################

def gen_param(lig):
    # parametrize ligand
    lig_par_folder=pocket_fit_folder+"/data/ligand/{}".format(lig.GetProp('ID'))
    ligand_itp=lig_par_folder+"/MOL.acpype/MOL_GMX.itp"
    if(not os.path.exists(ligand_itp)):

        if(not os.path.isdir(lig_par_folder)):
            os.makedirs(lig_par_folder)

        lig_pdb_file=lig_par_folder+"/MOL.pdb"
        with open(lig_pdb_file,"w") as f:
            f.write(rdmolfiles.MolToPDBBlock(lig, flavor=0))

        lig_file=lig_par_folder+"/MOL.mol"
        rdmolfiles.MolToMolFile(lig, lig_file)

        #find total charge
        totQ=rdmolops.GetFormalCharge(lig)

        #acpype
        os.chdir(lig_par_folder)
        with open('acpype.log', 'w') as acpype_log:
            process = subprocess.Popen(['acpype', '-i', lig_file, '-a', 'gaff', '-c', 'gas', '-b', "MOL", '-n', repr(totQ)],
                                    stdout=acpype_log, stderr=acpype_log)
            #stdout=subprocess.PIPE, stderr=subprocess.PIPE
            process.wait()

        os.chdir(tmpdir)
        

        

        
def setup_sim_folder_no_core(lig):
    #global tmpdir;
    #tmpdir=os.environ['TMPDIR']
    
    #create morphing folder
    #sim_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    out_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    sim_folder=f"{tmpdir}/{lig.GetProp('ID')}/"
    if(not os.path.isdir(sim_folder)):
        os.makedirs(sim_folder)

        #setup_folder=pocket_fit_folder+"/data/setup_energ_no_core/"
        setup_folder=tmpdir+"/setup_energ_no_core/"
        os.system("ln -s {} {}".format(setup_folder+"/*.itp", sim_folder+"/."))
        os.system("ln -s {} {}".format(setup_folder+"/*.pdb", sim_folder+"/."))
        os.system("ln -s {} {}".format(setup_folder+"/*.top", sim_folder+"/."))
    
    lig_par_folder=pocket_fit_folder+"/data/ligand/{}".format(lig.GetProp('ID'))
    lig_itp=lig_par_folder+"/MOL.acpype/MOL_GMX.itp"

    # copy ligand parameter file
    os.system(f"cp {lig_itp} {sim_folder}/.")

    #combine the protein and ligand structures
    os.chdir(sim_folder)
    lig_pdb_file=out_folder+"/morph_fit.pdb"
    os.system("tail -n +6 {} | cat prot.pdb - > combined.pdb".format(lig_pdb_file))


    #build index file
    os.system("echo -e 'q\n' | gmx make_ndx -f combined.pdb >/dev/null 2>&1".format(lig_pdb_file))
    if(not os.path.exists(sim_folder+"/index.ndx")):
        raise(Exception("make_ndx failed"))
    #add the a group containing only the ligand atoms, ligand+xray 
    #n = ndx.IndexFile(sim_folder+"/index.ndx", verbose=False)
    with suppress_stdout_stderr():
        n = ndx.IndexFile(sim_folder+"/index.ndx")
    shift = min(n["UNL"].ids) - 1
    grp = ndx.IndexGroup(ids=n["UNL"].ids, name="ligand")
    #grp = ndx.IndexGroup(ids=ligand_idxs, name="ligand")
    #n.add_group(grp, verbose=False)
    n.add_group(grp)
    
    #Find the no_core atom indeces
    core_smiles="c7(C)nc8ccccc8n8c(c6c(Cl)[cH][cH][cH][cH]6)nnc78"
    core=Chem.MolFromSmiles(core_smiles)
    
    with suppress_stdout_stderr():
        res,unmatched = rdRGD.RGroupDecompose([core], [lig], asSmiles=True)# print(unmatched)
    if(len(unmatched)>0):
        raise(Exception("Ligand does not contain core."))
        
    l=res[0]
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
        mapping=list(lig.GetSubstructMatch(Rgroup))
        
        #add to index
        nocore_index=[i+1+shift for i in mapping] # +1 to compensate for -1 in the shift
        grp = ndx.IndexGroup(ids=nocore_index, name="no_core_lig")
        n.add_group(grp)

    else:
        raise(Exception("Ligand is nothing but the core and hydrogens."))
    
    n.write(sim_folder+"/index.ndx")

    os.chdir(tmpdir)

        

    

def do_MDEnerg_no_core(lig, ncpus=1, redo=False):
    out_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    sim_folder=f"{tmpdir}/{lig.GetProp('ID')}/"
    setup_folder=f"{tmpdir}/setup_energ_no_core/"
    setup_folder_long_cut=f"{tmpdir}/setup_energ_long_cut/"
    setup_folder_no_core_long_cut=f"{tmpdir}/setup_energ_no_core_long_cut/"
    os.chdir(sim_folder)
    
    os.system(f"cat index.ndx {setup_folder}/close_prot_residues.ndx > index_for_energ_nocore.ndx")
    
    for i in range(1,4):
        if(not os.path.exists(out_folder+f"/energy_block_nocore_{i}.xvg") or redo):
            if(not os.path.exists(sim_folder+f"/energy_block_nocore_{i}.xvg")):
                os.system(f"gmx grompp -f {setup_folder}/lig_prot_energ_block{i}.mdp -c combined.pdb -o temp_lig_prot_energ_block{i}.tpr -n index_for_energ_nocore.ndx -p topol.top -maxwarn 3 > grompp_nocore_{i}.log 2>&1");
                os.system(f"mdrun_threads -ntomp {ncpus} -deffnm temp_lig_prot_energ_block{i} -rerun combined.pdb > /dev/null 2>&1")
                os.system(f"gmx energy -f temp_lig_prot_energ_block{i}.edr -o energy_block_nocore_{i}.xvg -xvg none < {setup_folder}/g_energy_block_{i}_input.txt > /dev/null 2>&1")
                if(not os.path.exists(f"energy_block_nocore_{i}.xvg")):
                    print(f"MDEnerg failed for nocore block {i}.")
                    raise(Exception(f"MDEnerg failed for nocore block {i}."))
                os.system(f"rm temp_lig_prot_energ_block{i}.*")
            os.system(f"cp energy_block_nocore_{i}.xvg {out_folder}/.")
            
    for i in range(1,4):
        if(not os.path.exists(out_folder+f"/energy_block_nocore_long_cut_{i}.xvg") or redo):
            if(not os.path.exists(sim_folder+f"/energy_block_nocore_long_cut_{i}.xvg")):
                os.system(f"gmx grompp -f {setup_folder_no_core_long_cut}/lig_prot_energ_block{i}.mdp -c combined.pdb -o temp_lig_prot_energ_block{i}.tpr -n index_for_energ_nocore.ndx -p topol.top -maxwarn 3 > grompp_nocore_long_cut_{i}.log 2>&1");
                os.system(f"mdrun_threads -ntomp {ncpus} -deffnm temp_lig_prot_energ_block{i} -rerun combined.pdb > /dev/null 2>&1")
                os.system(f"gmx energy -f temp_lig_prot_energ_block{i}.edr -o energy_block_nocore_long_cut_{i}.xvg -xvg none < {setup_folder_no_core_long_cut}/g_energy_block_{i}_input.txt > /dev/null 2>&1")
                if(not os.path.exists(f"energy_block_nocore_long_cut_{i}.xvg")):
                    print(f"MDEnerg failed for nocore_long_cut block {i}.")
                    raise(Exception(f"MDEnerg failed for nocore_long_cut block {i}."))
                os.system(f"rm temp_lig_prot_energ_block{i}.*")
            os.system(f"cp energy_block_nocore_long_cut_{i}.xvg {out_folder}/.")
            
    for i in range(1,4):
        if(not os.path.exists(out_folder+f"/energy_block_long_cut_{i}.xvg") or redo):
            if(not os.path.exists(sim_folder+f"/energy_block_long_cut_{i}.xvg")):
                os.system(f"gmx grompp -f {setup_folder_long_cut}/lig_prot_energ_block{i}.mdp -c combined.pdb -o temp_lig_prot_energ_block{i}.tpr -n index_for_energ_nocore.ndx -p topol.top -maxwarn 3 > grompp_long_cut_{i}.log 2>&1");
                os.system(f"mdrun_threads -ntomp {ncpus} -deffnm temp_lig_prot_energ_block{i} -rerun combined.pdb > /dev/null 2>&1")
                os.system(f"gmx energy -f temp_lig_prot_energ_block{i}.edr -o energy_block_long_cut_{i}.xvg -xvg none < {setup_folder_long_cut}/g_energy_block_{i}_input.txt > /dev/null 2>&1")
                if(not os.path.exists(f"energy_block_long_cut_{i}.xvg")):
                    print(f"MDEnerg failed for long_cut block {i}.")
                    raise(Exception(f"MDEnerg failed for long_cut block {i}."))
                os.system(f"rm temp_lig_prot_energ_block{i}.*")
            os.system(f"cp energy_block_long_cut_{i}.xvg {out_folder}/.")
            
    os.chdir(tmpdir)
    
    
# single call function
def do_ligand(lig_fn, redo=False, ncpus=1, clean=True):
    #reread $TMPDIR in case we run from notebook dirrectly
    global tmpdir;
    tmpdir=os.environ['TMPDIR']
    
    with open(lig_fn, 'rb') as f:
        lig = pickle.load(f)
    
    print(f"Processing {lig.GetProp('ID')} at {datetime.now()}", flush=True)
        
    #check if previously handled
    if(lig.HasProp('nocore MDenerg') and lig.GetProp('nocore MDenerg')=="yes" and not redo):
        print("ready")
        return()
        
    ##regenerate hydrogens
    #noH = rdmolops.RemoveHs(lig)
    #reH = rdmolops.AddHs(noH, addCoords=True)
    #prop_names = lig.GetPropNames()
    #for n in prop_names:
        #reH.SetProp(n, lig.GetProp(n))
    #lig = PropertyMol.PropertyMol(reH)
    #del prop_names, noH, reH;
        
    sim_folder=f"{tmpdir}/{lig.GetProp('ID')}/"
    setup_sim_folder_no_core(lig)
    print(f"Starting MDEnerg for {lig.GetProp('ID')} at {datetime.now()}", flush=True)
    do_MDEnerg_no_core(lig, ncpus, redo=redo)

    ##overwrite with new positions
    #new_mol=rdmolfiles.MolFromPDBFile(sim_folder+"/morph_fit.pdb", sanitize=False)
    #new_coords = new_mol.GetConformer().GetPositions()        
    #lig_conf = lig.GetConformer()
    #for j in range(lig.GetNumAtoms()):
        #lig_conf.SetAtomPosition(j, new_coords[j,:])

    lig.SetProp('nocore MDenerg', "yes")
    pickle.dump( PropertyMol.PropertyMol(lig), open( lig_fn, "wb" ) )
    
    #clean up TMPDIR
    if(clean):
        os.system(f"rm -fr {sim_folder}")
    
    print(f"Done with {lig.GetProp('ID')} at {datetime.now()}\n", flush=True)
    


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
    parser = argparse.ArgumentParser(description='Calculates per-residue prot-lig interaction energies while ignring the ligand core.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', dest='fname', type=str, nargs='+',
                       help='input/output pickle file')
    parser.add_argument('-n', dest='ncpus', type=int, default=1, help='number of cores to use with mdrun')
     
    parser.add_argument('--redo', dest='redo', type=str2bool, nargs='?', const=True, default=False,
                        help="Recalculate listed ligands, even if done previously.")
    
    args = parser.parse_args()

    #create a temp setup folder   
    setup_folder_src=pocket_fit_folder+"/data/setup_energ_no_core"
    setup_folder_dest=tmpdir+"/setup_energ_no_core"
    os.system(f"cp -r {setup_folder_src} {setup_folder_dest}")
    
    setup_folder_src=pocket_fit_folder+"/data/setup_energ_no_core_long_cut"
    setup_folder_dest=tmpdir+"/setup_energ_no_core_long_cut"
    os.system(f"cp -r {setup_folder_src} {setup_folder_dest}")
    
    setup_folder_src=pocket_fit_folder+"/data/setup_energ_long_cut"
    setup_folder_dest=tmpdir+"/setup_energ_long_cut"
    os.system(f"cp -r {setup_folder_src} {setup_folder_dest}")

    for f in args.fname:
        try:
            do_ligand(f, ncpus=args.ncpus)
        except Exception as e:
            sys.stderr.write(repr(e)+'\n')
            
    exit(0);
