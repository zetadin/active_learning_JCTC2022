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




try:
    import cPickle as pickle
except:
    import pickle 

pocket_fit_folder="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_4/morphing_annealing_4d09/"
tmpdir=os.environ['TMPDIR']

############################################################
# helper functions
############################################################

def find_mapping(mol, ref, outpath):    
#     #remove old output files
#     for f in os.listdir(outpath):
#         os.remove(os.path.join(outpath, f))
    
        
    #map atoms with pmx
    
    # params
    mol_file=outpath+"/mol.pdb"
    ref_file=outpath+"/ref.pdb"
    o1 = '{0}/ref_map.dat'.format(outpath)
    o2 = '{0}/mol_map.dat'.format(outpath)
    opdb1 = '{0}/out_pdb1.pdb'.format(outpath)
    opdb2 = '{0}/out_pdb2.pdb'.format(outpath)
    opdbm1 = '{0}/out_pdbm1.pdb'.format(outpath)
    opdbm2 = '{0}/out_pdbm2.pdb'.format(outpath)
    score = '{0}/score.dat'.format(outpath)
    log = '{0}/mapping.log'.format(outpath)
    
    #don't redo if output exists
    if (not(os.path.isfile(o2) and os.path.isfile(score))):
        
        #dump files
        with open(mol_file,"w") as f:
            f.write(rdmolfiles.MolToPDBBlock(mol))
        with open(ref_file,"w") as f:
            f.write(rdmolfiles.MolToPDBBlock(ref))

        process = subprocess.Popen(['pmx','atomMapping',
                            '-i1',ref_file,
                            '-i2',mol_file,
                            '-o1',o1,
                            '-o2',o2,
                            '-opdb1',opdb1,
                            '-opdb2',opdb2,                                        
                            '-opdbm1',opdbm1,
                            '-opdbm2',opdbm2,
                            '-score',score,
                            '-log',log,
                            '--dMCS', '--d', '0.5',
                            #'--RingsOnly'
                                   ],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        process.wait()

        if(not os.path.isfile(o2) ): #mapping failed, use less restrictive match criteria: no distance criterion in MCS
            process = subprocess.Popen(['pmx','atomMapping',
                            '-i1',ref_file,
                            '-i2',mol_file,
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


def find_lig_matches(ref, xray, matches):
    #find atom mappings
    with tempfile.TemporaryDirectory() as temp_dir:
        ref_match, xray_match, score = find_mapping(lig, xray, outpath = temp_dir)
    #ref.SetProp('score', str(score))
    
    
    return(ref_match, xray_match)


# collect atom types from both ligand and xray. pmx ligandHybrid  leaves out the [ atomtypes ] section
def get_atomtypes(fn):
    atomtypes={}
    with open(fn,"r") as f:
        in_atomtypes=False
        for l in f:
            line = l.strip()
            if not in_atomtypes:
                if "[ atomtypes ]" in line:
                    in_atomtypes=True
                    continue;
            else:
                if "[ " in line:
                    break; #end of atomtypes
                elif len(line)==0 or line[0]==';':
                    continue;  #empty or comment line
                else:
                    #c3       c3          0.00000  0.00000   A     3.39967e-01   4.57730e-01 ; 1.91  0.1094
                    s=line.split()
                    atomtypes[s[0]]=line
    return(atomtypes)    





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
        
        
def gen_pairs(lig, xray):
    if(lig.HasProp('# ligand new mapping') and  lig.HasProp('# xray new mapping')):
        lig_matches=eval(lig.GetProp('# ligand new mapping'))
        xray_matches=eval(lig.GetProp('# xray new mapping'))
    else:
        lig_par_folder=pocket_fit_folder+"/data/ligand/{}".format(lig.GetProp('ID'))
        morphing_top_folder=lig_par_folder+"/morphing_top"
        if(not os.path.isdir(morphing_top_folder)):
            os.makedirs(morphing_top_folder)
        lig_matches, xray_matches, score = find_mapping(lig, xray, morphing_top_folder)
        if(len(lig_matches)!=len(xray_matches)):
            raise(Exception("not all scaffold atoms found in ligand!"))
        #save so we don't have to redo this
        lig.SetProp('# ligand new mapping', lig_matches)
        lig.SetProp('# xray new mapping', xray_matches)
        lig.SetProp('score new mapping', score)
        
        
def gen_hybrid_top(lig, xray_name):
    #parametrize morphing topology
    lig_par_folder=pocket_fit_folder+"/data/ligand/{}".format(lig.GetProp('ID'))
    hybrid_folder=lig_par_folder+"/hybrid_top"
    hybrid_itp=hybrid_folder+"/merged.itp"
    if(not os.path.exists(hybrid_itp)):

        if(not os.path.isdir(hybrid_folder)):
            os.makedirs(hybrid_folder)


        lig_pdb=lig_par_folder+"/MOL.pdb"
        lig_itp=lig_par_folder+"/MOL.acpype/MOL_GMX.itp"

        xray_par_folder=pocket_fit_folder+"/data/xrays/{}".format(xray_name)
        xray_pdb=xray_par_folder+"/MOL.pdb"
        xray_itp=xray_par_folder+"/MOL.acpype/MOL_GMX.itp"

        pair_file=lig_par_folder+"/morphing_top/mol_map.dat"

        #make hybrid top
        os.chdir(hybrid_folder)
#         process = subprocess.Popen(['pmx','ligandHybrid', '-i1', lig_pdb, '-i2', xray_pdb,
#                                     '-itp1', lig_itp, '-itp2', xray_itp,
#                                     '-pairs', pair_file],
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE)
#         process.wait()
        cmd='pmx ligandHybrid -i1 '+ lig_pdb+ ' -i2 '+ xray_pdb+ ' -itp1 '+ lig_itp+ ' -itp2 '+ xray_itp + ' -pairs '+ pair_file
        os.system(cmd+" > hybrid_stdout.log 2>&1")
        
        if(not os.path.exists(hybrid_itp)):
            raise(Exception("pmx ligandHybrid failed"))

        os.chdir(tmpdir)
        
        
# collect atom types from both ligand and xray. pmx ligandHybrid  leaves out the [ atomtypes ] section
def get_atomtypes(fn):
    atomtypes={}
    with open(fn,"r") as f:
        in_atomtypes=False
        for l in f:
            line = l.strip()
            if not in_atomtypes:
                if "[ atomtypes ]" in line:
                    in_atomtypes=True
                    continue;
            else:
                if "[ " in line:
                    break; #end of atomtypes
                elif len(line)==0 or line[0]==';':
                    continue;  #empty or comment line
                else:
                    #c3       c3          0.00000  0.00000   A     3.39967e-01   4.57730e-01 ; 1.91  0.1094
                    s=line.split()
                    atomtypes[s[0]]=line
    return(atomtypes)

        
def setup_sim_folder(lig, xray_name):
    #create morphing folder
    #sim_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    out_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    sim_folder=f"{tmpdir}/{lig.GetProp('ID')}/"
    if(not os.path.isdir(sim_folder)):
        os.makedirs(sim_folder)

        #setup_folder=pocket_fit_folder+"/data/setup_morphes_sim_an/"
        setup_folder=tmpdir+"/setup_morphes_sim_an/"
        os.system("ln -s {} {}".format(setup_folder+"/*.*", sim_folder+"/."))


    #indeces of common atoms have been changed in the hybrid topology
    #find the new one from the hybrid.log:
    lig_par_folder=pocket_fit_folder+"/data/ligand/{}".format(lig.GetProp('ID'))
    hybrid_folder=lig_par_folder+"/hybrid_top"
    hybrid_log=hybrid_folder+"/hybrid.log"
    common_idxs=[]
    ligand_idxs=[]
    with open(hybrid_log,"r") as f:
        in_common=False
        in_dummiesB=False
        for line in f:
            if not in_common and "Making B-states...." in line:
                    in_common=True
                    continue;
            if in_common:
                if "Constructing dummies...." in line:
                    in_common=False
                    continue; #end of common atoms
                else:
                    #ligandHybridTop__log_> Atom....:   18            c3 |   0.02 |  12.01 ->           c3 |   0.04 |  12.01
                    s=line.split()
                    common_idxs.append(int(s[2]))
                    continue;

            if not in_dummiesB and "Dummies in stateB:" in line:
                    in_dummiesB=True
                    continue;
            if in_dummiesB:
                if "Construct bonds...." in line:
                    in_dummiesB=False
                    break; #end of dummiesB atoms, these come after common, so safe to break out of file reading loop.
                else:
                    #ligandHybridTop__log_> Dummy...:    1            c3 |  -0.04 |  12.01 ->       DUM_c3 |   0.00 |  12.01
                    s=line.split()
                    ligand_idxs.append(int(s[2]))
                    continue;

    ligand_idxs = common_idxs+ligand_idxs

    #build ligand restraint file
    MOL_posre_fn=sim_folder+"/MOL_posre.itp"
    with open(MOL_posre_fn, 'w') as f:
        f.write("[ position_restraints ]\n")
        f.write(";  i funct       fcx        fcy        fcz\n")
        for i in common_idxs:
            f.write("{}\t\t1       9000       9000       9000\n".format(i))



    #write hybrid atomtypes file
    xray_par_folder=pocket_fit_folder+"/data/xrays/{}".format(xray_name)
    xray_itp=xray_par_folder+"/MOL.acpype/MOL_GMX.itp"
    lig_par_folder=pocket_fit_folder+"/data/ligand/{}".format(lig.GetProp('ID'))
    lig_itp=lig_par_folder+"/MOL.acpype/MOL_GMX.itp"
    xray_ats=get_atomtypes(xray_itp)
    lig_ats=get_atomtypes(lig_itp)

    hybrid_atomtypes=sim_folder+"/hybrid_atomtypes.itp"
    with open(hybrid_atomtypes, 'w') as f:
        f.write("[ atomtypes ]\n")
        #print atomtypes from ligand
        for lig_key in lig_ats.keys():
            f.write(lig_ats[lig_key]+"\n")

        #print aditional atomtypes from xray
        for xray_key in xray_ats.keys():
            if(not xray_key in lig_ats.keys()):
                f.write(xray_ats[xray_key]+"\n")

    # copy ligand parameter file
    os.system("cp {}/*.itp {}/.".format(hybrid_folder, sim_folder))

    #combine the protein and ligand structures
    os.chdir(sim_folder)
    lig_pdb_file=hybrid_folder+"/mergedB.pdb"
    os.system("tail -n +3 {} | cat prot.pdb - > combined.pdb".format(lig_pdb_file))


    #build index file
    os.system("echo -e 'q\n' | gmx make_ndx -f combined.pdb >/dev/null 2>&1".format(lig_pdb_file))
    if(not os.path.exists(sim_folder+"/index.ndx")):
        raise(Exception("make_ndx failed"))
    #add the a group containing only the ligand atoms, ligand+xray 
    #n = ndx.IndexFile(sim_folder+"/index.ndx", verbose=False)
    n = ndx.IndexFile(sim_folder+"/index.ndx")
    shift = min(n["UNL"].ids) - 1
    for a in range(len(ligand_idxs)):
        ligand_idxs[a]+=shift
    grp = ndx.IndexGroup(ids=ligand_idxs, name="ligand")
    #n.add_group(grp, verbose=False)
    n.add_group(grp)
    n.write(sim_folder+"/index.ndx")

    os.chdir(tmpdir)

        
def do_sim(lig, ncpus=1):
    #run em for morphes
    #sim_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    out_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    sim_folder=f"{tmpdir}/{lig.GetProp('ID')}/"    
    os.chdir(sim_folder)

    #grompp
    if(not os.path.exists(sim_folder+"/em.tpr")):
        mdp_file=pocket_fit_folder+"/data/mdp/em_for_sim_an.mdp"
        cmd="gmx grompp -f {} -c combined.pdb -p topol.top -r combined.pdb -o em.tpr -maxwarn 4".format(mdp_file)
        os.system(cmd+" > grompp_em.log 2>&1")
        if(not os.path.exists(sim_folder+"/em.tpr")):
            raise(Exception("grompp for em failed"))

    #mdrun
    if(not os.path.exists(sim_folder+"/em.gro")):
        cmd="mdrun_threads -deffnm em -ntomp {} -ntmpi 1".format(ncpus)
        os.system(cmd+" > mdrun.log 2>&1")
        if(not os.path.exists(sim_folder+"/em.gro")):
            raise(Exception("em failed"))
        
    #grompp
    if(not os.path.exists(sim_folder+"/morph.tpr")):
        mdp_file=pocket_fit_folder+"/data/mdp/morphes_sim_an.mdp"
        cmd="gmx grompp -f {} -c em.gro -p topol.top -r combined.pdb -o morph.tpr -maxwarn 4".format(mdp_file)
        os.system(cmd+" > grompp_morph.log 2>&1")
        if(not os.path.exists(sim_folder+"/morph.tpr")):
            raise(Exception("grompp for morph failed"))

    #mdrun
    if(not os.path.exists(sim_folder+"/morph.gro")):
        cmd="mdrun_threads -deffnm morph -ntomp {} -ntmpi 1".format(ncpus)
        os.system(cmd+" > mdrun.log 2>&1")
        if(not os.path.exists(sim_folder+"/morph.gro")):
            raise(Exception("morphing failed"))

    #align back to initial positionsfix pbc and save only ligand
    if(not os.path.exists(sim_folder+"/morph_pbc.gro")):
        cmd="echo -e '15\n0\n' | gmx trjconv -s em.tpr -pbc mol -center -f morph.gro -o morph_pbc.gro"
        os.system(cmd+" > trjconv.log 2>&1")
        if(not os.path.exists(sim_folder+"/morph_pbc.gro")):
            raise(Exception("trjconv pbc+centering failed"))

    if(not os.path.exists(sim_folder+"/morph_fit.pdb")):
        cmd="echo -e 'backbone\nligand\n' | gmx trjconv -s em.tpr -fit rot+trans -f morph_pbc.gro -o morph_fit.pdb -n index.ndx"
        os.system(cmd+" >> trjconv.log 2>&1")
        if(not os.path.exists(sim_folder+"/morph_fit.pdb")):
            raise(Exception("trjconv fit failed"))
        
    if(not os.path.isdir(out_folder)):
        os.makedirs(out_folder)
    if(not os.path.exists(out_folder+"/morph_fit.pdb")):
        #os.system(f"cp {sim_folder}/morph.tpr {out_folder}/.")
        #os.system(f"cp {sim_folder}/hybrid_atomtypes.itp {out_folder}/.")
        #os.system(f"cp {sim_folder}/ffmerged.itp {out_folder}/.")
        #os.system(f"cp {sim_folder}/merged.itp {out_folder}/.")
        #os.system(f"cp {sim_folder}/MOL_posre.itp {out_folder}/.")
        
        #setup_folder_scr=pocket_fit_folder+"/data/setup_morphes_sim_an/"
        #os.system(f"ln -s {setup_folder_scr}/topol.top {out_folder}/.  >/dev/null 2>&1")
        #os.system(f"ln -s {setup_folder_scr}/pro*.itp {out_folder}/.  >/dev/null 2>&1")
        
        os.system(f"cp {sim_folder}/morph_fit.pdb {out_folder}/.")
            
    os.chdir(tmpdir)
    
    

def do_MDEnerg(lig, ncpus=1):
    out_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    sim_folder=f"{tmpdir}/{lig.GetProp('ID')}/"
    setup_folder=f"{tmpdir}/setup_energ/"
    os.chdir(sim_folder)
    
    os.system(f"cat index.ndx {setup_folder}/close_prot_residues.ndx > index_for_energ.ndx")
    
    for i in range(1,4):
        if(not os.path.exists(out_folder+f"/energy_block_{i}.xvg")):
            if(not os.path.exists(sim_folder+f"/energy_block_{i}.xvg")):
                os.system(f"gmx grompp -f {setup_folder}/lig_prot_energ_block{i}.mdp -c morph.gro -o temp_lig_prot_energ_block{i}.tpr -r combined.pdb  -n index_for_energ.ndx -p topol.top -maxwarn 3 > /dev/null 2>&1");
                os.system(f"mdrun_threads -ntomp {ncpus} -deffnm temp_lig_prot_energ_block{i} -rerun morph.gro > /dev/null 2>&1")
                os.system(f"gmx energy -f temp_lig_prot_energ_block{i}.edr -o energy_block_{i}.xvg -xvg none < {setup_folder}/g_energy_block_{i}_input.txt > /dev/null 2>&1")
                if(not os.path.exists(f"energy_block_{i}.xvg")):
                    raise(Exception(f"MDEnerg failed for block {i}."))
                os.system(f"rm temp_lig_prot_energ_block{i}.*")
            os.system(f"cp energy_block_{i}.xvg {out_folder}/.")
    os.chdir(tmpdir)
    
    
# single call function
def do_ligand(lig_fn, redo=False, ncpus=1):
    #reread $TMPDIR in case we run from notebook dirrectly
    global tmpdir;
    tmpdir=os.environ['TMPDIR']
    
    with open(lig_fn, 'rb') as f:
        lig = pickle.load(f)
    
    print(f"Processing {lig.GetProp('ID')} at {datetime.now()}", flush=True)
        
    #check if previously handled
    if(lig.HasProp('morphes done') and lig.GetProp('morphes done')=="yes" and not redo):
        print("ready")
        return()
        
    xray_name = lig.GetProp('xray_name')
    xray_fn = pocket_fit_folder+"/data/xrays/{0}/{0}.pickle".format(xray_name)
    with open(xray_fn, 'rb') as f:
        xray = pickle.load(f)
        
    #regenerate hydrogens
    noH = rdmolops.RemoveHs(lig)
    reH = rdmolops.AddHs(noH, addCoords=True)
    prop_names = lig.GetPropNames()
    for n in prop_names:
        reH.SetProp(n, lig.GetProp(n))
    lig = PropertyMol.PropertyMol(reH)
    del prop_names, noH, reH;
        
    gen_pairs(lig, xray)
    gen_param(lig)
    gen_hybrid_top(lig, xray_name)
    setup_sim_folder(lig, xray_name)
    print(f"Starting sims for {lig.GetProp('ID')} at {datetime.now()}", flush=True)
    do_sim(lig, ncpus)
    print(f"Starting MDEnerg for {lig.GetProp('ID')} at {datetime.now()}", flush=True)
    do_MDEnerg(lig, ncpus)

    #overwrite with new positions
    #sim_folder=pocket_fit_folder+"/morphes_sim_an/{}".format(lig.GetProp('ID'))
    sim_folder=f"{tmpdir}/{lig.GetProp('ID')}/"
    new_mol=rdmolfiles.MolFromPDBFile(sim_folder+"/morph_fit.pdb", sanitize=False)
    new_coords = new_mol.GetConformer().GetPositions()        
    lig_conf = lig.GetConformer()
    for j in range(lig.GetNumAtoms()):
        lig_conf.SetAtomPosition(j, new_coords[j,:])

    lig.SetProp('morphes done', "yes")
    pickle.dump( PropertyMol.PropertyMol(lig), open( lig_fn, "wb" ) )
    
    #clean up TMPDIR
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
    parser = argparse.ArgumentParser(description='Morphes xray structure into a ligand with simulated annealing to get a more reliable ligand structure.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', dest='fname', type=str, nargs='+',
                       help='input/output pickle file')
    parser.add_argument('-n', dest='ncpus', type=int, default=1, help='number of cores to use with mdrun')
     
    parser.add_argument('--redo', dest='redo', type=str2bool, nargs='?', const=True, default=False,
                        help="Recalculate listed ligands, even if done previously.")
    
    args = parser.parse_args()

    #create a temp setup folder
    setup_folder_src=pocket_fit_folder+"/data/setup_morphes_sim_an"
    setup_folder_dest=tmpdir+"/setup_morphes_sim_an"
    os.system(f"cp -r {setup_folder_src} {setup_folder_dest}")
    
    setup_folder_src=pocket_fit_folder+"/data/setup_energ"
    setup_folder_dest=tmpdir+"/setup_energ"
    os.system(f"cp -r {setup_folder_src} {setup_folder_dest}")

    for f in args.fname:
        try:
            do_ligand(f, ncpus=args.ncpus)
        except Exception as e:
            sys.stderr.write(repr(e)+'\n')
            
    exit(0);
