import pmx
from pmx.utils import create_folder
from pmx import gmx, ligand_alchemy, jobscript, ndx
import sys
import os,shutil
import re
import subprocess
import glob
import random
import pandas as pd
import numpy as np
import MDAnalysis as md
from copy import deepcopy
from multiprocessing import Process, Queue, Pool, TimeoutError, Lock
import dill

global mp_lock;
mp_lock = Lock()

#Packaging for multiprocessing according to https://stackoverflow.com/a/24673524
#This allows to call class functions via pool
def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    #mp_lock.acquire()
    #print(fun, args, *args)
    #mp_lock.release()
    return fun(*args)
def my_pool_apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))
def my_pool_map(pool, fun, args_list):
    #mp_lock.acquire()
    #print(args_list)
    #print(len(args_list))
    #mp_lock.release()
    payloads = [dill.dumps((fun, [arg])) for arg in args_list]
    return pool.map(run_dill_encoded, payloads)





class OptJobscript(pmx.jobscript.Jobscript):
    def __init__(self, **kwargs):
        self.acell_detect = """
            CPUFLAGS=$( cat /proc/cpuinfo | grep flags     | head -1 )
            VENDORID=$( cat /proc/cpuinfo | grep vendor_id | head -1 )

            if     [ $( echo $VENDORID | grep -c "AuthenticAMD" ) -eq "1" ]; then
                ACCEL="_AVX2_128"
            elif   [ $( echo $CPUFLAGS | grep -c "avx2"         ) -eq "1" ]; then
                ACCEL="_AVX2_256"
            elif   [ $( echo $CPUFLAGS | grep -c "avx"          ) -eq "1" ]; then
                ACCEL="_AVX_256"
            elif   [ $( echo $CPUFLAGS | grep -c "sse4_1"       ) -eq "1" ]; then
                ACCEL="_SSE4_1"
            elif   [ $( echo $CPUFLAGS | grep -c "sse2"         ) -eq "1" ]; then
                ACCEL="_SSE2"
            else
                ACCEL=""
            fi
            
        """
        self.simpath = ''        
        super(OptJobscript, self).__init__(**kwargs)
    
    def _create_header( self ):
        moduleline = ''
        sourceline = ''
        exportline = ''
        partitionline = self.partition
        for m in self.modules:
            moduleline = '{0}\nmodule load {1}'.format(moduleline,m)
        for s in self.source:
            sourceline = '{0}\nsource {1}'.format(sourceline,s)
        for e in self.export:
            exportline = '{0}\nexport load {1}'.format(exportline,e)
        
        exportline+='\n'+self.acell_detect
        
        gpuline = ''
        if self.bGPU==True:
            if self.queue == 'SGE':
                gpuline = '#$ -l gpu=1'
            elif self.queue == 'SLURM':
                gpuline = '#SBATCH --gres=gpu:1'
        gmxline = ''
        
        n_cpu=self.simcpu
        if(self.simcpu is None):
            if(self.queue == 'SLURM'):
                n_cpu="$SLURM_CPUS_ON_NODE"
            else:
                raise(Exception(f"SGE needs explicitly specified self.simcpu, but {self.simcpu} found."))
        
        
        if self.gmx!=None:
            if(not 'gmx' in self.gmx and "mdrun_threads" in self.gmx):
                gmxline = 'export GMXRUN="{gmx}$ACCEL -ntomp {simcpu} -ntmpi 1"'.format(gmx=self.gmx,simcpu=n_cpu)
            else:
                gmxline = 'export GMXRUN="{gmx} -ntomp {simcpu} -ntmpi 1"'.format(gmx=self.gmx,simcpu=n_cpu)            
            
        if self.queue=='SGE':
            self._create_SGE_header(moduleline,sourceline,exportline,gpuline,gmxline,partitionline)
        elif self.queue=='SLURM':
            self._create_SLURM_header(moduleline,sourceline,exportline,gpuline,gmxline,partitionline)

    
    # owerwrite parent's header to allow for auto CPU count
    def _create_SLURM_header( self,moduleline,sourceline,exportline,gpuline,gmxline,partitionline):
        fp = open(self.fname,'w')

        # optionally, can create a partition entry
        partition = ''
        if partitionline!=None and partitionline!='':
            partition = "#SBATCH --partition={0}\n".format(partitionline)

        self.header = '''#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --get-user-env
#SBATCH -N 1
{autocpu}
#SBATCH -t {simtime}:00:00
{partition}
{gpu}

{source}
{modules}
{export}

{gmx}
'''.format(jobname=self.jobname,
           autocpu= "" if self.simcpu is None else f"#SBATCH -n {self.simcpu}",
           simtime=self.simtime,partition=partition,
           gpu=gpuline,source=sourceline,modules=moduleline,export=exportline,
           gmx=gmxline)


class AZtutorial:
    """Class contains parameters for setting up free energy calculations

    Parameters
    ----------
    ...

    Attributes
    ----------
    ....

    """

    def __init__(self, **kwargs):
        
        # set gmxlib path
        gmx.set_gmxlib()
        
        # the results are summarized in a pandas framework
        self.resultsAll = pd.DataFrame()
        self.resultsSummary = pd.DataFrame()
        
        # paths
        self.workPath = './'
        self.mdpPath = '{0}/mdp'.format(self.workPath)
        self.proteinPath = None
        self.ligandPath = None
        
        # information about inputs
        self.protein = {} # protein[path]=path,protein[str]=pdb,protein[itp]=[itps],protein[posre]=[posres],protein[mols]=[molnames]
        self.ligands = {} # ligands[ligname]=path
        self.edges = {} # edges[edge_lig1_lig2] = [lig1,lig2]
        
        # parameters for the general setup
        self.replicas = 3        
        self.simTypes = ['em','eq','transitions']
        self.states = ['stateA','stateB']
        self.thermCycleBranches = ['water','protein']
                
        # simulation setup
        self.ff = 'amber99sb-star-ildn-mut.ff'
        self.boxshape = 'dodecahedron'
        self.boxd = 1.5
        self.water = 'tip3p'
        self.conc = 0.15
        self.pname = 'NaJ'
        self.nname = 'ClJ'
        
        # job submission params
        self.JOBqueue = 'SGE' # could be SLURM
        self.JOBsimtime = 24 # hours
        self.JOBsimcpu = 8 # CPU default
        self.JOBbGPU = True
        self.JOBmodules = []
        self.JOBsource = []
        self.JOBexport = []
        self.JOBgmx = 'gmx mdrun'
        self.JOBpartition = ''
        
        self.num_Xtal_waters = 0

        for key, val in kwargs.items():
            setattr(self,key,val)
            
    def prepareFreeEnergyDir( self, egde_suffix="" ):
        
        # protein={}
        # protein[path] = [path], protein[str] = pdb, protein[itp] = [itp], protein[posre] = [posre]
        self.proteinPath = self._read_path( self.proteinPath )
        self._protein = self._read_protein()
        
        # read ligands
        self.ligandPath = self._read_path( self.ligandPath )
        self._read_ligands()
        
        # read edges (directly or from a file)
        self._read_edges()
        
        # read mdpPath
        self.mdpPath = self._read_path( self.mdpPath )
        
        # workpath
        self.workPath = self._read_path( self.workPath )
        create_folder( self.workPath )
        
        # create folder structure
        self._create_folder_structure( egde_suffix=egde_suffix )
        
        # print summary
        self._print_summary( )
                        
        # print folder structure
        self._print_folder_structure( )    
        
        print('DONE')
        
        
    # _functions to quickly get a path at different levels, e.g wppath, edgepath... like in _create_folder_structure
    def _get_specific_path( self, edge=None, bHybridStrTop=False, wp=None, state=None, r=None, sim=None, egde_suffix="" ):
        if edge==None:
            return(self.workPath)       
        edgepath = '{0}/{1}{2}'.format(self.workPath,edge,egde_suffix)
        
        if bHybridStrTop==True:
            hybridStrPath = '{0}/hybridStrTop'.format(edgepath)
            return(hybridStrPath)

        if wp==None:
            return(edgepath)
        wppath = '{0}/{1}'.format(edgepath,wp)
        
        if state==None:
            return(wppath)
        statepath = '{0}/{1}'.format(wppath,state)
        
        if r==None:
            return(statepath)
        runpath = '{0}/run{1}'.format(statepath,r)
        
        if sim==None:
            return(runpath)
        simpath = '{0}/{1}'.format(runpath,sim)
        return(simpath)
                
    def _read_path( self, path ):
        return(os.path.abspath(path))
        
    def _read_ligands( self ):
        # read ligand folders
        ligs = glob.glob('{0}/*'.format(self.ligandPath))
        # get ligand names
        for l in ligs:
            lname = l.split('/')[-1]
            lnameTrunc = lname
            if lname.startswith('lig_'):
                lnameTrunc = lname[4:]
            elif lname.startswith('lig'):
                lnameTrunc = lname[3:]
            lpath = '{0}/{1}'.format(self.ligandPath,lname)
            self.ligands[lnameTrunc] = os.path.abspath(lpath)
 
    def _read_protein( self ):
        # read protein folder
        self.protein['path'] = os.path.abspath(self.proteinPath)
        # get folder contents
        self.protein['posre'] = []
        self.protein['itp'] = []
        self.protein['mols'] = [] # mols to add to .top
        self.protein['str'] = ''
        for l in glob.glob('{0}/*'.format(self.proteinPath)):
            fname = l.split('/')[-1]
            if '.itp' in fname: # posre or top
                if 'posre' in fname:
                    self.protein['posre'].append(os.path.abspath(l))
                else:
                    self.protein['itp'].append(os.path.abspath(l))
                    if fname.startswith('topol_'):
                        self.protein['mols'].append(fname[6:-4])
                    else:
                        self.protein['mols'].append(fname[:-4])                        
            if '.pdb' in fname:
                self.protein['str'] = fname
        self.protein['mols'].sort()
                
    def _read_edges( self ):
        # read from file
        try:
            if os.path.isfile( self.edges ):
                self._read_edges_from_file( self )
        # edge provided as an array
        except: 
            foo = {}
            for e in self.edges:
                key = 'edge_{0}_{1}'.format(e[0],e[1])
                foo[key] = e
            self.edges = foo
            
    def _read_edges_from_file( self ):
        self.edges = 'Edges read from file'
        
        
    def _create_folder_structure( self, edges=None, egde_suffix="" ):
        # edge
        if edges==None:
            edges = self.edges        
        for edge in edges:
            print(edge)            
            edgepath = '{0}/{1}{2}'.format(self.workPath,edge, egde_suffix)
            create_folder(edgepath)
            
            # folder for hybrid ligand structures
            hybridTopFolder = '{0}/hybridStrTop'.format(edgepath)
            create_folder(hybridTopFolder)
            
            # water/protein
            for wp in self.thermCycleBranches:
                wppath = '{0}/{1}'.format(edgepath,wp)
                create_folder(wppath)
                
                # stateA/stateB
                for state in self.states:
                    statepath = '{0}/{1}'.format(wppath,state)
                    create_folder(statepath)
                    
                    # run1/run2/run3
                    for r in range(1,self.replicas+1):
                        runpath = '{0}/run{1}'.format(statepath,r)
                        create_folder(runpath)
                        
                        # em/eq_posre/eq/transitions
                        for sim in self.simTypes:
                            simpath = '{0}/{1}'.format(runpath,sim)
                            create_folder(simpath)
                            
    def _print_summary( self ):
        print('\n---------------------\nSummary of the setup:\n---------------------\n')
        print('   workpath: {0}'.format(self.workPath))
        print('   mdp path: {0}'.format(self.mdpPath))
        print('   protein files: {0}'.format(self.proteinPath))
        print('   ligand files: {0}'.format(self.ligandPath))
        print('   number of replicase: {0}'.format(self.replicas))        
        print('   edges:')
        for e in self.edges.keys():
            print('        {0}'.format(e))    
            
    def _print_folder_structure( self ):
        print('\n---------------------\nDirectory structure:\n---------------------\n')
        print('{0}/'.format(self.workPath))
        print('|')
        print('|--edge_X_Y')
        print('|--|--water')
        print('|--|--|--stateA')
        print('|--|--|--|--run1/2/3')
        print('|--|--|--|--|--em/eq_posre/eq/transitions')        
        print('|--|--|--stateB')
        print('|--|--|--|--run1/2/3')
        print('|--|--|--|--|--em/eq_posre/eq/transitions')       
        print('|--|--protein')
        print('|--|--|--stateA')
        print('|--|--|--|--run1/2/3')
        print('|--|--|--|--|--em/eq_posre/eq/transitions')        
        print('|--|--|--stateB')
        print('|--|--|--|--run1/2/3')
        print('|--|--|--|--|--em/eq_posre/eq/transitions')    
        print('|--|--hybridStrTop')        
        print('|--edge_..')
        
    def _be_verbose( self, process, bVerbose=False, bShowErr=True, input=None ):
        out = process.communicate(input)
        if bVerbose==True:
            printout = out[0].splitlines()
            for o in printout:
                print(o)
        # error is printed every time (by default, but acan be overriden)
        if(bShowErr):
            printerr = out[1].splitlines()
            for e in printerr:
                print(e)
        
    def atom_mapping( self, edges=None, bVerbose=False, n_processes=1 ):
        print('-----------------------')
        print('Performing atom mapping')
        print('-----------------------')
        
        def atom_mapping_Single(edge):
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            
            lig1 = self.edges[edge][0]
            lig2 = self.edges[edge][1]
            lig1path = '{0}/lig_{1}'.format(self.ligandPath,lig1)
            lig2path = '{0}/lig_{1}'.format(self.ligandPath,lig2)
            outpath = self._get_specific_path(edge=edge,bHybridStrTop=True)
            
            # params
            #i1 = '{0}/mol_gmx.pdb'.format(lig1path)
            #i2 = '{0}/mol_gmx.pdb'.format(lig2path)
            i1 = '{0}/mol_sigmahole.pdb'.format(lig1path)
            i2 = '{0}/mol_sigmahole.pdb'.format(lig2path)
            o1 = '{0}/pairs1.dat'.format(outpath)
            o2 = '{0}/pairs2.dat'.format(outpath)            
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
                                '-log',log],
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)

            self._be_verbose( process, bVerbose=bVerbose )
            process.wait()
        
        if edges==None:
            edges = self.edges
        if(n_processes==1):
            for edge in edges:        
                atom_mapping_Single(edge)
        elif(n_processes>1):
            with Pool(processes=n_processes, initargs=(mp_lock,)) as pool:
                #res=pool.map(atom_mapping_Single, list(edges.keys()))
                res=my_pool_map(pool, atom_mapping_Single, list(edges.keys()))
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))
        print('DONE')            
            
            
    def hybrid_structure_topology( self, edges=None, bVerbose=False, n_processes=1, DummyMassScale=1.0 ):
        print('----------------------------------')
        print('Creating hybrid structure/topology')
        print('----------------------------------')
        
        def hybrid_structure_topology_Single(edge):
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            
            lig1 = self.edges[edge][0]
            lig2 = self.edges[edge][1]
            lig1path = '{0}/lig_{1}'.format(self.ligandPath,lig1)
            lig2path = '{0}/lig_{1}'.format(self.ligandPath,lig2)
            outpath = self._get_specific_path(edge=edge,bHybridStrTop=True)
            
            # params
            #i1 = '{0}/mol_gmx.pdb'.format(lig1path)
            #i2 = '{0}/mol_gmx.pdb'.format(lig2path)
            #itp1 = '{0}/MOL.itp'.format(lig1path)
            #itp2 = '{0}/MOL.itp'.format(lig2path)   
            i1 = '{0}/mol_sigmahole.pdb'.format(lig1path)
            i2 = '{0}/mol_sigmahole.pdb'.format(lig2path)
            itp1 = '{0}/MOL_sigmahole.itp'.format(lig1path)
            itp2 = '{0}/MOL_sigmahole.itp'.format(lig2path)            
            pairs = '{0}/pairs1.dat'.format(outpath)            
            oA = '{0}/mergedA.pdb'.format(outpath)
            oB = '{0}/mergedB.pdb'.format(outpath)
            oitp = '{0}/merged.itp'.format(outpath)
            offitp = '{0}/ffmerged.itp'.format(outpath)
            log = '{0}/hybrid.log'.format(outpath)
            
            process = subprocess.Popen(['pmx','ligandHybrid',
                                '-i1',i1,
                                '-i2',i2,
                                '-itp1',itp1,
                                '-itp2',itp2,
                                '-pairs',pairs,
                                '-oA',oA,  
                                '-oB',oB,
                                '-oitp',oitp,
                                '-offitp',offitp,
                                '--scDUMm',str(DummyMassScale),
                                '-log',log],
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)

            self._be_verbose( process, bVerbose=bVerbose )                    
                
            process.wait()    
        if edges==None:
            edges = self.edges        
        if(n_processes==1):
            for edge in edges:        
                hybrid_structure_topology_Single(edge)
        elif(n_processes>1):
            raise(Exception("hybrid_structure_topology() does not support n_processes>1"))
            #with Pool(processes=n_processes) as pool:
                #res=my_pool_map(pool, hybrid_structure_topology_Single, list(edges.keys()))
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))
        print('DONE')


    def hybrid_structure_restraints( self, edges=None, bVerbose=False, n_processes=1  ):
        print('-----------------------------------------------')
        print('Creating position restraints for hybrid ligands')
        print('-----------------------------------------------')
    
        def hybrid_structure_restraints_Single(edge):
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            
            outpath = self._get_specific_path(edge=edge,bHybridStrTop=True)
            atom_type_columns={'A':1, 'B':8}
            for s in ['A','B']:
                # params
                structf = '{0}/merged{1}.pdb'.format(outpath,s)
                ndxf = '{0}/index{1}.ndx'.format(outpath,s)
                posref = '{0}/posre{1}.itp'.format(outpath,s)
                itp = '{0}/merged.itp'.format(outpath)
                
                # make the index file
                #process = subprocess.Popen(['gmx','make_ndx',
                                    #'-f',structf,
                                    #'-o',ndxf],
                                    #stdin= subprocess.PIPE,
                                    #stdout=subprocess.PIPE, 
                                    #stderr=subprocess.PIPE)

                
                #make_ndx_cmd="a H*\na D*\n!3&!4\n\nq\n"
                ##make_ndx_cmd="q\n"
                #self._be_verbose( process, bVerbose=bVerbose, input=make_ndx_cmd.encode(), bShowErr=False )
                #process.wait()
                
                heavy_non_dummies=[]
                with open(itp, 'r') as f:
                    lines=f.readlines()
                    b_in_atoms=False
                    for l in lines:
                        if(l[0]==';'):
                            continue;
                        elif(not b_in_atoms):
                            if "[ atoms ]" in l:
                                b_in_atoms=True
                                continue;
                        else:
                            if "[ " in l:
                                b_in_atoms=False
                                break;
                            else:
                                sp=l.split()
                                try:
                                    if(len(sp)>0 and ( (len(sp)==8 and not "DUM" in l ) or not "DUM" in sp[atom_type_columns[s]] )):
                                        if(sp[4][0]!='H'):
                                            heavy_non_dummies.append(int(sp[0]))
                                except Exception as e:
                                    print(sp)
                                    raise(e)
                            
                            #elif(not "DUM" in l):
                                #sp=l.split()
                                #if(len[sp]>0):
                                    #heavy_non_dummies.append(int(sp[0]))
                
                ndx_file = ndx.IndexFile()
                g = ndx.IndexGroup(ids=heavy_non_dummies, name="heavy_non_dummies")
                ndx_file.add_group(g)
                ndx_file.write(ndxf)
                            
                
                #make the restraints
                process = subprocess.Popen(['gmx','genrestr',
                                    '-f',structf,
                                    '-fc', '1000', '1000', '1000',
                                    '-n',ndxf,
                                    '-o', posref],
                                    stdin= subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                
                genrestr_cmd="heavy_non_dummies\n"
                self._be_verbose( process, bVerbose=bVerbose, input=genrestr_cmd.encode(), bShowErr=False )
                process.wait()

        if edges==None:
            edges = self.edges        
        if(n_processes==1):
            for edge in edges:        
                hybrid_structure_restraints_Single(edge)
        elif(n_processes>1):
            with Pool(processes=n_processes) as pool:
                #res=pool.map(hybrid_structure_restraints_Single, list(edges.keys()))
                res=my_pool_map(pool, hybrid_structure_restraints_Single, list(edges.keys()))
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))
        print('DONE')
            
            
    def _make_clean_pdb(self, fnameIn,fnameOut,bAppend=False):
        # read 
        fp = open(fnameIn,'r')
        lines = fp.readlines()
        out = []
        for l in lines:
            if l.startswith('ATOM') or l.startswith('HETATM'):
                out.append(l)
        fp.close()
        
        # write
        if bAppend==True:
            fp = open(fnameOut,'a')
        else:
            fp = open(fnameOut,'w')
        for l in out:
            fp.write(l)
        fp.close()
            
    def assemble_systems( self, edges=None, filter_prot_water=False, prot_extra_mols={}):
        print('----------------------')
        print('Assembling the systems')
        print('----------------------')

        if edges==None:
            edges = self.edges        
        for edge in edges:
            print(edge)            
            lig1 = self.edges[edge][0]
            lig2 = self.edges[edge][1]
            lig1path = '{0}/lig_{1}'.format(self.ligandPath,lig1)
            lig2path = '{0}/lig_{1}'.format(self.ligandPath,lig2)
            hybridStrTopPath = self._get_specific_path(edge=edge,bHybridStrTop=True)                    
            outLigPath = self._get_specific_path(edge=edge,wp='water')
            outProtPath = self._get_specific_path(edge=edge,wp='protein')
                        
            # Ligand structure
            self._make_clean_pdb('{0}/mergedA.pdb'.format(hybridStrTopPath),'{0}/init.pdb'.format(outLigPath))
            
            # Ligand+Protein structure
            self._make_clean_pdb('{0}/{1}'.format(self.proteinPath,self.protein['str']),'{0}/init.pdb'.format(outProtPath))
            self._make_clean_pdb('{0}/mergedA.pdb'.format(hybridStrTopPath),'{0}/init.pdb'.format(outProtPath),bAppend=True)
            
            self.num_Xtal_waters=0
            m_prot_extra_mols=deepcopy(prot_extra_mols)
            if(filter_prot_water):
                u_init = md.Universe(f'{outProtPath}/init.pdb')
                not_water = u_init.select_atoms('not resname HOH')
                ligand = u_init.select_atoms('resname UNL')
                if(len(ligand.atoms)==0):
                    raise(Exception("Could not find ligand UNL. Check ligand name in pdbs."))
                water = u_init.select_atoms('resname HOH')
                if(len(water.atoms)==0):
                    raise(Exception("There is no water in the system to filter."))
                water_too_close = u_init.select_atoms('byres (resname HOH and name OW) and around 2.0 resname UNL')
                if(len(water_too_close.atoms)>0): # otherwize there is nothing to filter
                    print("changing number of waters to ", end="")
                    os.rename(f'{outProtPath}/init.pdb', f'{outProtPath}/~init.pdb')
                    water_good = u_init.select_atoms('resname HOH and not group water_too_close', water_too_close=water_too_close)
                    filtered_init = u_init.select_atoms('group not_water or group water_good', not_water=not_water, water_good=water_good)
                    
                    filtered_init.write(f'{outProtPath}/filtered_init.pdb')
                    water_too_close.write(f'{outProtPath}/removed_water.pdb')
                    self._make_clean_pdb(f'{outProtPath}/filtered_init.pdb', f'{outProtPath}/init.pdb')
                    
                    self.num_Xtal_waters=len(water_good.residues)
                    if("SOL" in list(m_prot_extra_mols.keys())):
                        m_prot_extra_mols["SOL"]=self.num_Xtal_waters
                        print(self.num_Xtal_waters)
                
    
            # Ligand topology
            # ffitp
            ffitpOut = '{0}/ffmerged.itp'.format(hybridStrTopPath)
            #ffitpIn1 = '{0}/ffMOL.itp'.format(lig1path)
            #ffitpIn2 = '{0}/ffMOL.itp'.format(lig2path)
            ffitpIn1 = '{0}/ffMOL_sigmahole.itp'.format(lig1path)
            ffitpIn2 = '{0}/ffMOL_sigmahole.itp'.format(lig2path)
            ffitpIn3 = '{0}/ffmerged.itp'.format(hybridStrTopPath)
            pmx.ligand_alchemy._merge_FF_files( ffitpOut, ffsIn=[ffitpIn1,ffitpIn2,ffitpIn3] )        
            # top        
            ligTopFname = '{0}/topol.top'.format(outLigPath)
            ligFFitp = '{0}/ffmerged.itp'.format(hybridStrTopPath)
            ligItp ='{0}/merged.itp'.format(hybridStrTopPath)
            itps = [ligFFitp,ligItp]
            posres = [None, None]
            systemName = 'ligand in water'
            self._create_top(fname=ligTopFname,itp=itps,systemName=systemName, posres=posres)
            
            # Ligand+Protein topology
            # top
            for state in ['A','B']:
                protTopFname = '{0}/topol{1}.top'.format(outProtPath, state)
                mols = []
                for m in self.protein['mols']:
                    mols.append([m,1])
                for key in m_prot_extra_mols.keys():
                    mols.append([key, m_prot_extra_mols[key]])
                mols.append(['MOL',1])
                itps_p=itps + self.protein['itp']
                posres = [None, f"{hybridStrTopPath}/posre{state}.itp"] + self.protein['posre']
                systemName = 'protein and ligand in water'
                self._create_top(fname=protTopFname,itp=itps_p,mols=mols,systemName=systemName, posres=posres)            
        print('DONE')            
        
            
    def _create_top( self, fname='topol.top',  
                   itp=['merged.itp'], mols=[['MOL',1]],
                   systemName='simulation system',
                   destination='',toppaths=[], posres=[None] ):

        fp = open(fname,'w')
        # ff itp
        fp.write('#include "%s/forcefield.itp"\n' % self.ff)
        # additional itp
        if(len(posres)!=len(itp)):
            raise(Exception(f"Number of posres files does not match number of molecular itp files:\n{itp}\n{posres}"))
        for i,it in enumerate(itp):
            fp.write('#include "%s"\n' % it)
            #add position restraints
            if(not posres[i] is None):
                fp.write('#ifdef POSRES\n')
                fp.write('#include "%s"\n' % posres[i])
                fp.write('#endif\n')

        # water itp
        fp.write('#include "%s/%s.itp"\n' % (self.ff,self.water)) 
        # ions
        fp.write('#include "%s/ions.itp"\n\n' % self.ff)
        # system
        fp.write('[ system ]\n')
        fp.write('{0}\n\n'.format(systemName))
        # molecules
        fp.write('[ molecules ]\n')
        for mol in mols:
            fp.write('%s %s\n' %(mol[0],mol[1]))
        fp.close()

        
    def _clean_backup_files( self, path ):
        toclean = glob.glob('{0}/*#'.format(path)) 
        for clean in toclean:
            os.remove(clean)        
    
    def boxWaterIons( self, edges=None, bBoxLig=True, bBoxProt=True, 
                                        bWatLig=True, bWatProt=True,
                                        bIonLig=True, bIonProt=True,
                                        n_processes=1):
        print('----------------')
        print('Box, water, ions')
        print('----------------')
        
        def boxWaterIons_Single(tup):
            edge, bBoxLig, bBoxProt, bWatLig, bWatProt, bIonLig, bIonProt = tup;
            
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            
            outLigPath = self._get_specific_path(edge=edge,wp='water')
            outProtPath = self._get_specific_path(edge=edge,wp='protein')
            
            # box ligand
            if bBoxLig==True:
                inStr = '{0}/init.pdb'.format(outLigPath)
                outStr = '{0}/box.pdb'.format(outLigPath)
                gmx.editconf(inStr, o=outStr, bt=self.boxshape, d=self.boxd, other_flags='')                
            # box protein
            if bBoxProt==True:            
                inStr = '{0}/init.pdb'.format(outProtPath)
                outStr = '{0}/box.pdb'.format(outProtPath)
                gmx.editconf(inStr, o=outStr, bt=self.boxshape, d=self.boxd, other_flags='')
                
            # water ligand
            if bWatLig==True:            
                inStr = '{0}/box.pdb'.format(outLigPath)
                outStr = '{0}/water.pdb'.format(outLigPath)
                top = '{0}/topol.top'.format(outLigPath)
                gmx.solvate(inStr, cs='spc216.gro', p=top, o=outStr)
            # water protein
            if bWatProt==True:            
                inStr = '{0}/box.pdb'.format(outProtPath)
                outStr = '{0}/water.pdb'.format(outProtPath)
                top = '{0}/topolA.top'.format(outProtPath)
                gmx.solvate(inStr, cs='spc216.gro', p=top, o=outStr)
            
            # ions ligand
            if bIonLig:
                inStr = '{0}/water.pdb'.format(outLigPath)
                outStr = '{0}/ions.pdb'.format(outLigPath)
                mdp = '{0}/em_l0.mdp'.format(self.mdpPath)
                tpr = '{0}/tpr.tpr'.format(outLigPath)
                top = '{0}/topol.top'.format(outLigPath)
                mdout = '{0}/mdout.mdp'.format(outLigPath)
                gmx.grompp(f=mdp, c=inStr, p=top, o=tpr, maxwarn=4, other_flags=' -po {0} > {1}/genion.log 2>&1'.format(mdout, outLigPath))        
                gmx.genion(s=tpr, p=top, o=outStr, conc=self.conc, neutral=True, 
                      other_flags=' -pname {0} -nname {1} >> {2}/genion.log 2>&1'.format(self.pname, self.nname, outLigPath))
            # ions protein
            if bIonProt:
                inStr = '{0}/water.pdb'.format(outProtPath)
                outStr = '{0}/ions.pdb'.format(outProtPath)
                mdp = '{0}/em_l0.mdp'.format(self.mdpPath)
                tpr = '{0}/tpr.tpr'.format(outProtPath)
                top = '{0}/topolA.top'.format(outProtPath)
                mdout = '{0}/mdout.mdp'.format(outProtPath)
                index = '{0}/genion.ndx'.format(outProtPath)
                
                #create an index file
                os.system(f"echo 'q\n' | gmx make_ndx -f {inStr} -o {index} > {outProtPath}/genion.log 2>&1")
                #clean it and separate SOL into crystal and non-crystal groups
                ndx_file = ndx.IndexFile(index)

                xtal_sol_ids=[ i for i in ndx_file["SOL"].ids if i<ndx_file["UNL"].ids[0] ]
                g = ndx.IndexGroup(ids=xtal_sol_ids, name="xtal_SOL")
                ndx_file.add_group(g)

                sol_ids=[ i for i in ndx_file["SOL"].ids if i>ndx_file["UNL"].ids[0] ]
                g = ndx.IndexGroup(ids=sol_ids, name="SOL")
                ndx_file.delete_group("SOL")
                ndx_file.add_group(g)

                ndx_file.write(index)
                
                #make the ions
                gmx.grompp(f=mdp, c=inStr, p=top, o=tpr, maxwarn=4, other_flags=' -po {0} -n {1}  >> {2}/genion.log 2>&1'.format(mdout, index, outProtPath))        
                gmx.genion(s=tpr, p=top, o=outStr, conc=self.conc, neutral=True, 
                      other_flags=' -pname {0} -nname {1} -n {2}  >> {3}/genion.log 2>&1'.format(self.pname, self.nname, index, outProtPath))
                
                #copy water & ions from topolA to topolB
                os.system(f"sed -n '/MOL /,$p' {outProtPath}/topolA.top | tail -n +2 >> {outProtPath}/topolB.top")
                
            # clean backed files
            self._clean_backup_files( outLigPath )
            self._clean_backup_files( outProtPath )
        
        if edges==None:
            edges = self.edges
            
        if(n_processes==1):
            for edge in edges:
                boxWaterIons_Single((edge, bBoxLig, bBoxProt, bWatLig, bWatProt, bIonLig, bIonProt))
        elif(n_processes>1):
            args=[(edge, bBoxLig, bBoxProt, bWatLig, bWatProt, bIonLig, bIonProt) for edge in edges]
            with Pool(processes=n_processes) as pool:
                #res=pool.map(boxWaterIons_Single, args)
                res=my_pool_map(pool, boxWaterIons_Single, args)
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))

        print('DONE')
            
    def _prepare_single_tpr( self, simpath, toppath, state, simType, empath=None, eqposrepath=None, frameNum=0, b_prot=False ):
        
        mdpPrefix = ''
        extra_flags=""
        if simType=='em':
            mdpPrefix = 'em'
        elif simType=='eq_posre':
            mdpPrefix = 'eq_posre'
        elif simType=='eq':
            mdpPrefix = 'eq'
        elif simType=='transitions':
            mdpPrefix = 'ti'        
            
        top = '{0}/topol.top'.format(toppath)
        if(b_prot):
            top = '{0}/topol{1}.top'.format(toppath, state[-1:])
        tpr = '{0}/tpr.tpr'.format(simpath)
        mdout = '{0}/mdout.mdp'.format(simpath)
        # mdp
        if state=='stateA':
            mdp = '{0}/{1}_l0.mdp'.format(self.mdpPath,mdpPrefix)
        else:
            mdp = '{0}/{1}_l1.mdp'.format(self.mdpPath,mdpPrefix)
        # str
        if simType=='em':
            inStr = '{0}/ions.pdb'.format(toppath)
        elif simType=='eq_posre':
            inStr = '{0}/confout.gro'.format(empath)
        elif simType=='eq':
            inStr = '{0}/confout.gro'.format(eqposrepath)
            #inStr = '{0}/confout.gro'.format(empath)
            #extra_flags += "-t {0}/state.cpt".format(eqposrepath)
        elif simType=='transitions':
            inStr = '{0}/frame{1}.gro'.format(simpath,frameNum)
            tpr = '{0}/ti{1}.tpr'.format(simpath,frameNum)
            
        extra_flags+=f" > {simpath}/grompp.log 2>&1"
        
        if(not os.path.isfile(tpr)):
            gmx.grompp(f=mdp, c=inStr, p=top, o=tpr, maxwarn=4, other_flags=' -po {0} {1}'.format(mdout, extra_flags))
            self._clean_backup_files( simpath )
            if(not os.path.isfile(tpr)):
                raise(Exception(f"Failed to generate tpr: {tpr}"));
                    
         
    def prepare_simulation( self, edges=None, simType='em', bLig=True, bProt=True, n_processes=1, egde_suffix="" ):
        print('-----------------------------------------')
        print('Preparing simulation: {0}'.format(simType))
        print('-----------------------------------------')
        
        mdpPrefix = ''
        if simType=='em':
            mdpPrefix = 'em'
        elif simType=='eq_posre':
            mdpPrefix = 'eq_posre'
        elif simType=='eq':
            mdpPrefix = 'eq'
        elif simType=='transitions':
            mdpPrefix = 'ti'
            
        def prepare_simulation_Single(edge):
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            
            ligTopPath = self._get_specific_path(edge=edge,wp='water', egde_suffix=egde_suffix)
            protTopPath = self._get_specific_path(edge=edge,wp='protein', egde_suffix=egde_suffix)            
            
            for state in self.states:
                for r in range(1,self.replicas+1):
                    
                    # ligand
                    if bLig==True:
                        wp = 'water'
                        simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim=simType, egde_suffix=egde_suffix)
                        empath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='em', egde_suffix=egde_suffix)
                        #eqposrepath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='eq_posre')
                        toppath = ligTopPath
                        self._prepare_single_tpr( simpath, toppath, state, simType, empath, eqposrepath=empath, b_prot=False )
                    
                    # protein
                    if bProt==True:
                        wp = 'protein'
                        simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim=simType, egde_suffix=egde_suffix)
                        empath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='em', egde_suffix=egde_suffix)
                        eqposrepath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='eq_posre', egde_suffix=egde_suffix)
                        toppath = protTopPath
                        self._prepare_single_tpr( simpath, toppath, state, simType, empath, eqposrepath, b_prot=True )   
                        
        if edges==None:
            edges = self.edges
        if(n_processes==1):
            for edge in edges:
                prepare_simulation_Single(edge)
        elif(n_processes>1):
            with Pool(processes=n_processes) as pool:
                #res=pool.map(prepare_simulation_Single, list(edges.keys()))
                res=my_pool_map(pool, prepare_simulation_Single, list(edges.keys()))
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))
        

    def _run_mdrun( self, tpr=None, ener=None, confout=None, mdlog=None, 
                    cpo=None, trr=None, xtc=None, dhdl=None, bVerbose=False,
                    mdrun=["gmx", "mdrun"] ):
        # EM
        if xtc==None:
            process = subprocess.Popen(mdrun+[
                                '-s',tpr,
                                '-e',ener,
                                '-c',confout,
                                '-o',trr,                                        
                                '-g',mdlog],
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
            self._be_verbose( process, bVerbose=bVerbose )                    
            process.wait()           
        # other FE runs
        else:
            process = subprocess.Popen(mdrun+[
                                '-s',tpr,
                                '-e',ener,
                                '-c',confout,
                                '-dhdl',dhdl,
                                '-x',xtc,
                                '-o',trr,
                                '-cpo',cpo,                                        
                                '-g',mdlog],
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
            self._be_verbose( process, bVerbose=bVerbose )                    
            process.wait()           
            

    def run_simulation_locally( self, edges=None, simType='em', bLig=True, bProt=True, bVerbose=False, mdrun=["gmx", "mdrun"] ):
        print('-------------------------------------------')
        print('Run simulation locally: {0}'.format(simType))
        print('-------------------------------------------')
        
        if edges==None:
            edges = self.edges
        for edge in edges:
            
            for state in self.states:
                for r in range(1,self.replicas+1):            
                    
                    # ligand
                    if bLig==True:
                        wp = 'water'
                        print('Running: LIG {0} {1} run{2}'.format(edge,state,r))
                        simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim=simType)
                        tpr = '{0}/tpr.tpr'.format(simpath)
                        ener = '{0}/ener.edr'.format(simpath)
                        confout = '{0}/confout.gro'.format(simpath)
                        mdlog = '{0}/md.log'.format(simpath)
                        trr = '{0}/traj.trr'.format(simpath)                        
                        self._run_mdrun(tpr=tpr,trr=trr,ener=ener,confout=confout,mdlog=mdlog,bVerbose=bVerbose, mdrun=mdrun)
                        self._clean_backup_files( simpath )
                    
                    # protein
                    if bProt==True:
                        wp = 'protein'
                        print('Running: PROT {0} {1} run{2}'.format(edge,state,r))
                        simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim=simType)
                        tpr = '{0}/tpr.tpr'.format(simpath)
                        ener = '{0}/ener.edr'.format(simpath)
                        confout = '{0}/confout.gro'.format(simpath)
                        mdlog = '{0}/md.log'.format(simpath)
                        trr = '{0}/traj.trr'.format(simpath)                                                
                        self._run_mdrun(tpr=tpr,trr=trr,ener=ener,confout=confout,mdlog=mdlog,bVerbose=bVerbose, mdrun=mdrun)
                        self._clean_backup_files( simpath )
        print('DONE')
 
    def prepare_jobscripts( self, edges=None, simType='em', bLig=True, bProt=True, job_folder_suffix="", egde_suffix=""):
        print('---------------------------------------------')
        print('Preparing jobscripts for: {0}'.format(simType))
        print('---------------------------------------------')
        
        continue_detect="""
if [ -f "{simpath}/state.cpt" ]; then
    CONT="-cpi {simpath}/state.cpt"
else
    CONT=""
fi

"""
        
        jobfolder = '{0}/{1}_jobscripts{2}'.format(self.workPath,simType,job_folder_suffix)
        os.system('mkdir {0}'.format(jobfolder))
        
        if edges==None:
            edges = self.edges
            
        counter = 0
        for edge in edges:
            
            for state in self.states:
                for r in range(1,self.replicas+1):            
                    
                    # ligand
                    if bLig==True:
                        wp = 'water'
                        simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim=simType, egde_suffix=egde_suffix)
                        if(not os.path.isfile(simpath+"/confout.gro") ):
                            jobfile = '{0}/jobscript{1}'.format(jobfolder,counter)
                            jobname = 'lig_{0}_{1}_{2}_{3}'.format(edge,state,r,simType)
                            job = OptJobscript(fname=jobfile,
                                            queue=self.JOBqueue,simcpu=self.JOBsimcpu,simtime=self.JOBsimtime,
                                            jobname=jobname,modules=self.JOBmodules,source=self.JOBsource,
                                            gmx=self.JOBgmx, bGPU=self.JOBbGPU, simpath=simpath, partition=self.JOBpartition)

                            cmd1 = 'cd {0}'.format(simpath)
                            cmd2 = '$GMXRUN -s tpr.tpr $CONT'
                            job.cmds = [continue_detect.format(simpath=simpath),cmd1,cmd2]
                                                       
                            b_make_script=True
                            if simType=='transitions':
                                dHdl_files = glob.glob('{0}/*xvg'.format(simpath))
                                if(len(dHdl_files)<80):
                                    eqpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='eq', egde_suffix=egde_suffix)
                                    self._commands_for_transitions( simpath, job, eqpath, wp=wp,state=state )
                                else:
                                    b_make_script=False
                            if(b_make_script):
                                job.create_jobscript()
                                counter+=1
                    
                    # protein
                    if bProt==True:
                        wp = 'protein'
                        simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim=simType, egde_suffix=egde_suffix)
                        if(not os.path.isfile(simpath+"/confout.gro") ):
                            jobfile = '{0}/jobscript{1}'.format(jobfolder,counter)
                            jobname = 'prot_{0}_{1}_{2}_{3}'.format(edge,state,r,simType)
                            job = OptJobscript(fname=jobfile,
                                            queue=self.JOBqueue,simcpu=self.JOBsimcpu,simtime=self.JOBsimtime,
                                            jobname=jobname,modules=self.JOBmodules,source=self.JOBsource,
                                            gmx=self.JOBgmx, simpath=simpath, partition=self.JOBpartition,
                                            cont_check=(simType!='transitions'))
                            cmd1 = 'cd {0}'.format(simpath)
                            cmd2 = '$GMXRUN -s tpr.tpr $CONT'
                            job.cmds = [continue_detect.format(simpath=simpath),cmd1,cmd2]
                            
                            b_make_script=True
                            if simType=='transitions':
                                dHdl_files = glob.glob('{0}/*xvg'.format(simpath))
                                if(len(dHdl_files)<80):
                                    eqpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='eq', egde_suffix=egde_suffix)
                                    self._commands_for_transitions( simpath, job, eqpath, wp=wp,state=state )
                                else:
                                    b_make_script=False
                            if(b_make_script):
                                job.create_jobscript()
                                counter+=1
                        
        #######
        self._submission_script( jobfolder, counter, simType )
        print('DONE')
        
    def _commands_for_transitions( self, simpath, job, source_path, wp, state ):
        for i in range(1,81):
            if self.JOBqueue=='SGE':
                #cmd1 = 'cd $TMPDIR'
                #cmd2 = 'cp {0}/ti$SGE_TASK_ID.tpr tpr.tpr'.format(simpath)
                #cmd3 = '$GMXRUN -s tpr.tpr -dhdl dhdl$SGE_TASK_ID.xvg'.format(simpath)
                #cmd4 = 'cp dhdl$SGE_TASK_ID.xvg {0}/.'.format(simpath)
                #job.cmds = [cmd1,cmd2,cmd3,cmd4]
                job.cmds=[f"""
                export GMXLIB={os.environ['GMXLIB']}
                                        
                cd $TMPDIR

                # Copy the data to TMPDIR
                cp -r $GMXLIB/amber99sb-star-ildn-mut.ff .
                cp -r {self.mdpPath}/ti*.mdp .
                """]
                
                topname="topol.top"
                if(wp=='protein'):
                    if(state=='stateA'):
                        topname="topolA.top"
                    else:
                        topname="topolB.top"

                job.cmds[0]+=f"""
    
                for f in $(sed -E -n '/amber99sb/! s/^#include \\"(.*)\\"/\\1/p' {simpath}/../../../{topname})
                do
                cp $f .
                done

                sed -E '/amber99sb/! s/^#include \\".*\/(.*\.itp)\\"/#include \\"\\1\\"/g' {simpath}/../../../{topname} > {topname}
                """
                                   
                mdpname=""
                if(state=='stateA'):
                    mdpname="ti_l0.mdp"
                else:
                    mdpname="ti_l1.mdp"
                    
                job.cmds[0]+=f"""
                # Dump gro files
                echo -e "0\\n" | gmx trjconv -s {source_path}/tpr.tpr -f {source_path}/traj.trr -o frame.gro -pbc mol -ur compact -b 2250 -sep
                mv frame0.gro frame80.gro > {simpath}/grompp.log 2>&1

                for i in {{1..80}};do
                    if [[ ! -s {simpath}/dhdl$i.xvg ]]; then
                        gmx grompp -f {mdpname} -c frame$i.gro -p {topname} -o ti$i.tpr -maxwarn 4 >> {simpath}/grompp.log 2>&1
                        $GMXRUN -deffnm ti$i -s ti$i.tpr -dhdl dhdl$i
                        cp dhdl$i.xvg {simpath}/.
                        # clean up after the sim (running out of space otherwize)
                        rm ti$i.* dhdl$i.xvg frame$i.gro core* *.pdb
                    else
                        echo '{simpath}/dhdl$i.xvg already finished.'
                    fi
                done
                """
            elif self.JOBqueue=='SLURM':
                #job.cmds=[f"""
                #cd $TMPDIR
                #for i in {{1..81}};do
                    #$GMXRUN -deffnm ti$i -s {simpath}/ti$i.tpr -dhdl dhdl$i
                    #cp dhdl$i.xvg {simpath}/.
                #done
                #"""]
                job.cmds=[f"""
                export GMXLIB={os.environ['GMXLIB']}
                                        
                cd $TMPDIR

                # Copy the data to TMPDIR
                cp -r $GMXLIB/amber99sb-star-ildn-mut.ff .
                cp -r {self.mdpPath}/ti*.mdp .
                """]
                
                topname="topol.top"
                if(wp=='protein'):
                    if(state=='stateA'):
                        topname="topolA.top"
                    else:
                        topname="topolB.top"

                job.cmds[0]+=f"""
    
                for f in $(sed -E -n '/amber99sb/! s/^#include \\"(.*)\\"/\\1/p' {simpath}/../../../{topname})
                do
                cp $f .
                done

                sed -E '/amber99sb/! s/^#include \\".*\/(.*\.itp)\\"/#include \\"\\1\\"/g' {simpath}/../../../{topname} > {topname}
                """
                                   
                mdpname=""
                if(state=='stateA'):
                    mdpname="ti_l0.mdp"
                else:
                    mdpname="ti_l1.mdp"
                    
                job.cmds[0]+=f"""
                # Dump gro files
                echo -e "0\\n" | gmx trjconv -s {source_path}/tpr.tpr -f {source_path}/traj.trr -o frame.gro -pbc mol -ur compact -b 2250 -sep
                mv frame0.gro frame80.gro > {simpath}/grompp.log 2>&1

                for i in {{1..80}};do
                    if [[ ! -s {simpath}/dhdl$i.xvg ]]; then
                        gmx grompp -f {mdpname} -c frame$i.gro -p {topname} -o ti$i.tpr -maxwarn 4 >> {simpath}/grompp.log 2>&1
                        $GMXRUN -deffnm ti$i -s ti$i.tpr -dhdl dhdl$i
                        cp dhdl$i.xvg {simpath}/.
                        # clean up after the sim (running out of space otherwize)
                        rm ti$i.* dhdl$i.xvg frame$i.gro core* *.pdb
                    else
                        echo '{simpath}/dhdl$i.xvg already finished.'
                    fi
                done
                """

        
        
    def _submission_script( self, jobfolder, counter, simType='eq' ):
        fname = '{0}/submit.py'.format(jobfolder)
        fp = open(fname,'w')
        fp.write('import os\n')
        fp.write('for i in range(0,{0}):\n'.format(counter))
        if self.JOBqueue=='SGE':
            cmd = '\'qsub jobscript{0}\'.format(i)'
            if simType=='transitions':
                #cmd = '\'qsub -t 1-80:1 jobscript{0}\'.format(i)'
                cmd = '\'qsub jobscript{0}; sleep 5;\'.format(i)' # wait 5 s between submissions to avoid all jobs starting at once and overloading the network again.
        elif self.JOBqueue=='SLURM':
            cmd = '\'sbatch jobscript{0}\'.format(i)'
        fp.write('    os.system({0})\n'.format(cmd))
        fp.close()

    def _extract_snapshots( self, eqpath, tipath ):
        if(not os.path.exists(tipath+"/frame80.gro")):
            tpr = '{0}/tpr.tpr'.format(eqpath)
            trr = '{0}/traj.trr'.format(eqpath)
            frame = '{0}/frame.gro'.format(tipath)
            
            gmx.trjconv(s=tpr,f=trr,o=frame, sep=True, ur='compact', pbc='mol', other_flags=' -b 2250')
            # move frame0.gro to frame80.gro
            cmd = 'mv {0}/frame0.gro {0}/frame80.gro'.format(tipath)
            os.system(cmd)
        
        self._clean_backup_files( tipath )
        
        
    def prepare_transitions( self, edges=None, bLig=True, bProt=True, bGenTpr=True ):
        print('---------------------')
        print('Preparing transitions')
        print('---------------------')
        
        if edges==None:
            edges = self.edges
        for edge in edges:
            ligTopPath = self._get_specific_path(edge=edge,wp='water')
            protTopPath = self._get_specific_path(edge=edge,wp='protein')            
            
            for state in self.states:
                for r in range(1,self.replicas+1):
                    
                    # ligand
                    if bLig==True:
                        print('Preparing: LIG {0} {1} run{2}'.format(edge,state,r))
                        wp = 'water'
                        eqpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='eq')
                        tipath = simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='transitions')
                        toppath = ligTopPath
                        self._extract_snapshots( eqpath, tipath )
                        if bGenTpr==True:
                            for i in range(1,81):
                                self._prepare_single_tpr( tipath, toppath, state, simType='transitions',frameNum=i, b_prot=False )
                    
                    # protein
                    if bProt==True:
                        print('Preparing: PROT {0} {1} run{2}'.format(edge,state,r))
                        wp = 'protein'
                        eqpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='eq')
                        tipath = simpath = self._get_specific_path(edge=edge,wp=wp,state=state,r=r,sim='transitions')                        
                        toppath = protTopPath
                        self._extract_snapshots( eqpath, tipath )
                        if bGenTpr==True:
                            for i in range(1,81):
                                self._prepare_single_tpr( tipath, toppath, state, simType='transitions',frameNum=i, b_prot=True )
        print('DONE')  
        
        
    def _run_analysis_script( self, analysispath, stateApath, stateBpath, bVerbose=False, only_check_completition=False ):
        gA = glob.glob('{0}/*xvg'.format(stateApath))
        gB = glob.glob('{0}/*xvg'.format(stateBpath))
        if(only_check_completition):
            if(len(gA)>=80):
                #print(stateApath, "\tReady")
                pass;
            else:
                print(stateApath, f"\tNot ready, only {len(gA)} dHdl files found.")
            if(len(gB)>=80):
                #print(stateApath, "\tReady")
                pass;
            else:
                print(stateBpath, f"\tNot ready, only {len(gB)} dHdl files found.")
            return
        fA = ' '.join( gA )
        fB = ' '.join( gB )
        oA = '{0}/integ0.dat'.format(analysispath)
        oB = '{0}/integ1.dat'.format(analysispath)
        wplot = '{0}/wplot.png'.format(analysispath)
        o = '{0}/results.txt'.format(analysispath)
        
        # check if previous analysis already finished
        prevA = False;
        if(os.path.isfile(oA)):
            with open(oA, 'r') as fp:
                lines = fp.readlines()
                if(len(lines)>=80):
                    prevA=True
        prevB = False;
        if(os.path.isfile(oB)):
            with open(oB, 'r') as fp:
                lines = fp.readlines()
                if(len(lines)>=80):
                    prevB=True
        
        if( not (prevA and prevB) ):
            # if not previously fully completed, run analysis
            cmd = 'pmx analyse -fA {0} -fB {1} -o {2} -oA {3} -oB {4} -w {5} -t {6} -b {7}'.format(fA,fB,o,oA,oB,wplot,298,100) 
            os.system(cmd)
        
        if bVerbose==True:
            fp = open(o,'r')
            lines = fp.readlines()
            fp.close()
            bPrint = False
            for l in lines:
                if 'ANALYSIS' in l:
                    bPrint=True
                if bPrint==True:
                    print(l,end='')
        
    def run_analysis( self, edges=None, bLig=True, bProt=True, bParseOnly=False, bVerbose=False, only_check_completition=False, n_processes=1 ):
        print('----------------')
        print('Running analysis')
        print('----------------')
        
        def run_analysis_Single(edge):
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            for r in range(1,self.replicas+1):
                # ligand
                if bLig==True:
                    wp = 'water'
                    analysispath = '{0}/analyse{1}'.format(self._get_specific_path(edge=edge,wp=wp),r)
                    create_folder(analysispath)
                    stateApath = self._get_specific_path(edge=edge,wp=wp,state='stateA',r=r,sim='transitions')
                    stateBpath = self._get_specific_path(edge=edge,wp=wp,state='stateB',r=r,sim='transitions')
                    self._run_analysis_script( analysispath, stateApath, stateBpath, bVerbose=bVerbose, only_check_completition=only_check_completition )
                    
                # protein
                if bProt==True:
                    wp = 'protein'
                    analysispath = '{0}/analyse{1}'.format(self._get_specific_path(edge=edge,wp=wp),r)
                    create_folder(analysispath)
                    stateApath = self._get_specific_path(edge=edge,wp=wp,state='stateA',r=r,sim='transitions')
                    stateBpath = self._get_specific_path(edge=edge,wp=wp,state='stateB',r=r,sim='transitions')
                    self._run_analysis_script( analysispath, stateApath, stateBpath, bVerbose=bVerbose, only_check_completition=only_check_completition )
                    
        if edges==None:
            edges = self.edges
        if(n_processes==1):
            for edge in edges:        
                run_analysis_Single(edge)
        elif(n_processes>1):
            with Pool(processes=n_processes, initargs=(mp_lock,)) as pool:
                res=my_pool_map(pool, run_analysis_Single, list(edges.keys()))
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))
        print('DONE')
        
        
    def _read_neq_results( self, fname ):
        fp = open(fname,'r')
        lines = fp.readlines()
        fp.close()
        out = []
        for l in lines:
            l = l.rstrip()
            foo = l.split()
            if 'BAR: dG' in l:
                out.append(float(foo[-2]))
            elif 'BAR: Std Err (bootstrap)' in l:
                out.append(float(foo[-2]))
            elif 'BAR: Std Err (analytical)' in l:
                out.append(float(foo[-2]))      
            elif '0->1' in l:
                out.append(int(foo[-1]))      
            elif '1->0' in l:
                out.append(int(foo[-1]))
        return(out)         
    
    def _fill_resultsAll( self, res, edge, wp, r ):
        rowName = '{0}_{1}_{2}'.format(edge,wp,r)
        self.resultsAll.loc[rowName,'val'] = res[2]
        self.resultsAll.loc[rowName,'err_analyt'] = res[3]
        self.resultsAll.loc[rowName,'err_boot'] = res[4]
        self.resultsAll.loc[rowName,'framesA'] = res[0]
        self.resultsAll.loc[rowName,'framesB'] = res[1]
        
    def _summarize_results( self, edges ):
        bootnum = 1000
        for edge in edges:
            for wp in ['water','protein']:
                dg = []
                erra = []
                errb = []
                distra = []
                distrb = []
                for r in range(1,self.replicas+1):
                    rowName = '{0}_{1}_{2}'.format(edge,wp,r)
                    dg.append( self.resultsAll.loc[rowName,'val'] )
                    erra.append( self.resultsAll.loc[rowName,'err_analyt'] )
                    errb.append( self.resultsAll.loc[rowName,'err_boot'] )
                    distra.append(np.random.normal(self.resultsAll.loc[rowName,'val'],self.resultsAll.loc[rowName,'err_analyt'] ,size=bootnum))
                    distrb.append(np.random.normal(self.resultsAll.loc[rowName,'val'],self.resultsAll.loc[rowName,'err_boot'] ,size=bootnum))
                  
                rowName = '{0}_{1}'.format(edge,wp)
                distra = np.array(distra).flatten()
                distrb = np.array(distrb).flatten()

                if self.replicas==1:
                    self.resultsAll.loc[rowName,'val'] = dg[0]                              
                    self.resultsAll.loc[rowName,'err_analyt'] = erra[0]
                    self.resultsAll.loc[rowName,'err_boot'] = errb[0]
                else:
                    self.resultsAll.loc[rowName,'val'] = np.mean(dg)
                    self.resultsAll.loc[rowName,'err_analyt'] = np.sqrt(np.var(distra)/float(self.replicas))
                    self.resultsAll.loc[rowName,'err_boot'] = np.sqrt(np.var(distrb)/float(self.replicas))
                    
            #### also collect resultsSummary
            rowNameWater = '{0}_{1}'.format(edge,'water')
            rowNameProtein = '{0}_{1}'.format(edge,'protein')            
            dg = self.resultsAll.loc[rowNameProtein,'val'] - self.resultsAll.loc[rowNameWater,'val']
            erra = np.sqrt( np.power(self.resultsAll.loc[rowNameProtein,'err_analyt'],2.0) \
                            + np.power(self.resultsAll.loc[rowNameWater,'err_analyt'],2.0) )
            errb = np.sqrt( np.power(self.resultsAll.loc[rowNameProtein,'err_boot'],2.0) \
                            + np.power(self.resultsAll.loc[rowNameWater,'err_boot'],2.0) )
            rowName = edge
            self.resultsSummary.loc[rowName,'val'] = dg
            self.resultsSummary.loc[rowName,'err_analyt'] = erra
            self.resultsSummary.loc[rowName,'err_boot'] = errb
            
                    
    def analysis_summary( self, edges=None ):
        if edges==None:
            edges = self.edges
            
        for edge in edges:
            for r in range(1,self.replicas+1):
                for wp in ['water','protein']:
                    try:
                        analysispath = '{0}/analyse{1}'.format(self._get_specific_path(edge=edge,wp=wp),r)
                        resultsfile = '{0}/results.txt'.format(analysispath)
                        res = self._read_neq_results( resultsfile )
                        self._fill_resultsAll( res, edge, wp, r )
                    except Exception as e:
                        print("Problem with:", resultsfile)
                        raise(e)
        
        # the values have been collected now
        # let's calculate ddGs
        self._summarize_results( edges )






    ###############################################
    ########end state only#########################

    def end_state_structure_restraints( self, edges=None, bVerbose=False, n_processes=1  ):
        print('-----------------------------------------------')
        print('Creating position restraints for end state ligands')
        print('-----------------------------------------------')
    
        def end_state_structure_restraints_Single(edge):
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            
            lig2 = self.edges[edge][1]
            lig2path = '{0}/lig_{1}'.format(self.ligandPath,lig2)
            
            atom_type_columns={'A':1, 'B':8}
            for s in ['B']:
                # params
                structf = '{0}/mol_sigmahole.pdb'.format(lig2path)
                ndxf = '{0}/index{1}.ndx'.format(lig2path,s)
                posref = '{0}/posre{1}.itp'.format(lig2path,s)
                itp = '{0}/MOL_sigmahole.itp'.format(lig2path)
                               
                heavy_non_dummies=[]
                with open(itp, 'r') as f:
                    lines=f.readlines()
                    b_in_atoms=False
                    for l in lines:
                        if(l[0]==';'):
                            continue;
                        elif(not b_in_atoms):
                            if "[ atoms ]" in l:
                                b_in_atoms=True
                                continue;
                        else:
                            if "[ " in l:
                                b_in_atoms=False
                                break;
                            else:
                                sp=l.split()
                                try:
                                    if(len(sp)>0 and ( (len(sp)==8 and not "DUM" in l ) or not "DUM" in sp[atom_type_columns[s]] )):
                                        if(sp[4][0]!='H'):
                                            heavy_non_dummies.append(int(sp[0]))
                                except Exception as e:
                                    print(sp)
                                    raise(e)
                
                ndx_file = ndx.IndexFile()
                g = ndx.IndexGroup(ids=heavy_non_dummies, name="heavy_non_dummies")
                ndx_file.add_group(g)
                ndx_file.write(ndxf)
                            
                
                #make the restraints
                process = subprocess.Popen(['gmx','genrestr',
                                    '-f',structf,
                                    '-fc', '1000', '1000', '1000',
                                    '-n',ndxf,
                                    '-o', posref],
                                    stdin= subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                
                genrestr_cmd="heavy_non_dummies\n"
                self._be_verbose( process, bVerbose=bVerbose, input=genrestr_cmd.encode(), bShowErr=False )
                process.wait()

        if edges==None:
            edges = self.edges        
        if(n_processes==1):
            for edge in edges:        
                end_state_structure_restraints_Single(edge)
        elif(n_processes>1):
            with Pool(processes=n_processes) as pool:
                res=my_pool_map(pool, end_state_structure_restraints_Single, list(edges.keys()))
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))
        print('DONE')
        
    def end_state_assemble_systems( self, edges=None, filter_prot_water=False, prot_extra_mols={}):
        print('----------------------')
        print('Assembling the systems')
        print('----------------------')

        if edges==None:
            edges = self.edges        
        for edge in edges:
            print(edge)            
            #lig1 = self.edges[edge][0]
            lig2 = self.edges[edge][1]
            #lig1path = '{0}/lig_{1}'.format(self.ligandPath,lig1)
            lig2path = '{0}/lig_{1}'.format(self.ligandPath,lig2)
            #hybridStrTopPath = self._get_specific_path(edge=edge,bHybridStrTop=True)                    
            outLigPath = self._get_specific_path(edge=edge,wp='water', egde_suffix="_end_state")
            outProtPath = self._get_specific_path(edge=edge,wp='protein', egde_suffix="_end_state")
                        
            # Ligand structure
            self._make_clean_pdb('{0}/mol_sigmahole.pdb'.format(lig2path),'{0}/init.pdb'.format(outLigPath))
            
            # Ligand+Protein structure
            self._make_clean_pdb('{0}/{1}'.format(self.proteinPath,self.protein['str']),'{0}/init.pdb'.format(outProtPath))
            self._make_clean_pdb('{0}/mol_sigmahole.pdb'.format(lig2path),'{0}/init.pdb'.format(outProtPath),bAppend=True)
            
            self.num_Xtal_waters=0
            m_prot_extra_mols=deepcopy(prot_extra_mols)
            if(filter_prot_water):
                u_init = md.Universe(f'{outProtPath}/init.pdb')
                not_water = u_init.select_atoms('not resname HOH')
                ligand = u_init.select_atoms('resname UNL')
                if(len(ligand.atoms)==0):
                    raise(Exception("Could not find ligand UNL. Check ligand name in pdbs."))
                water = u_init.select_atoms('resname HOH')
                if(len(water.atoms)==0):
                    raise(Exception("There is no water in the system to filter."))
                water_too_close = u_init.select_atoms('byres (resname HOH and name OW) and around 2.0 resname UNL')
                if(len(water_too_close.atoms)>0): # otherwize there is nothing to filter
                    print("changing number of waters to ", end="")
                    os.rename(f'{outProtPath}/init.pdb', f'{outProtPath}/~init.pdb')
                    water_good = u_init.select_atoms('resname HOH and not group water_too_close', water_too_close=water_too_close)
                    filtered_init = u_init.select_atoms('group not_water or group water_good', not_water=not_water, water_good=water_good)
                    
                    filtered_init.write(f'{outProtPath}/filtered_init.pdb')
                    water_too_close.write(f'{outProtPath}/removed_water.pdb')
                    self._make_clean_pdb(f'{outProtPath}/filtered_init.pdb', f'{outProtPath}/init.pdb')
                    
                    self.num_Xtal_waters=len(water_good.residues)
                    if("SOL" in list(m_prot_extra_mols.keys())):
                        m_prot_extra_mols["SOL"]=self.num_Xtal_waters
                        print(self.num_Xtal_waters)
                
    
            # Ligand topology
            # ffitp
            #ffitpOut = '{0}/ffmerged.itp'.format(hybridStrTopPath)
            #ffitpIn1 = '{0}/ffMOL_sigmahole.itp'.format(lig1path)
            #ffitpIn2 = '{0}/ffMOL_sigmahole.itp'.format(lig2path)
            #ffitpIn3 = '{0}/ffmerged.itp'.format(hybridStrTopPath)
            #pmx.ligand_alchemy._merge_FF_files( ffitpOut, ffsIn=[ffitpIn1,ffitpIn2,ffitpIn3] )        
            # top        
            ligTopFname = '{0}/topol.top'.format(outLigPath)
            ligFFitp = '{0}/ffMOL_sigmahole.itp'.format(lig2path)
            ligItp ='{0}/MOL_sigmahole.itp'.format(lig2path)
            itps = [ligFFitp,ligItp]
            posres = [None, None]
            systemName = 'ligand in water'
            self._create_top(fname=ligTopFname,itp=itps,systemName=systemName, posres=posres)
            
            # Ligand+Protein topology
            # top
            for state in ['B']:
                protTopFname = '{0}/topol{1}.top'.format(outProtPath, state)
                mols = []
                for m in self.protein['mols']:
                    mols.append([m,1])
                for key in m_prot_extra_mols.keys():
                    mols.append([key, m_prot_extra_mols[key]])
                mols.append(['MOL',1])
                itps_p=itps + self.protein['itp']
                posres = [None, f"{lig2path}/posre{state}.itp"] + self.protein['posre']
                systemName = 'protein and ligand in water'
                self._create_top(fname=protTopFname,itp=itps_p,mols=mols,systemName=systemName, posres=posres)            
        print('DONE')            
        
    def end_state_boxWaterIons( self, edges=None, bBoxLig=True, bBoxProt=True, 
                                        bWatLig=True, bWatProt=True,
                                        bIonLig=True, bIonProt=True,
                                        n_processes=1):
        print('----------------')
        print('Box, water, ions')
        print('----------------')
        
        def end_state_boxWaterIons_Single(tup):
            edge, bBoxLig, bBoxProt, bWatLig, bWatProt, bIonLig, bIonProt = tup;
            
            mp_lock.acquire()
            print(edge)
            mp_lock.release()
            
            outLigPath = self._get_specific_path(edge=edge,wp='water', egde_suffix="_end_state")
            outProtPath = self._get_specific_path(edge=edge,wp='protein', egde_suffix="_end_state")
            
            # box ligand
            if bBoxLig==True:
                inStr = '{0}/init.pdb'.format(outLigPath)
                outStr = '{0}/box.pdb'.format(outLigPath)
                gmx.editconf(inStr, o=outStr, bt=self.boxshape, d=self.boxd, other_flags='')                
            # box protein
            if bBoxProt==True:            
                inStr = '{0}/init.pdb'.format(outProtPath)
                outStr = '{0}/box.pdb'.format(outProtPath)
                gmx.editconf(inStr, o=outStr, bt=self.boxshape, d=self.boxd, other_flags='')
                
            # water ligand
            if bWatLig==True:            
                inStr = '{0}/box.pdb'.format(outLigPath)
                outStr = '{0}/water.pdb'.format(outLigPath)
                top = '{0}/topol.top'.format(outLigPath)
                gmx.solvate(inStr, cs='spc216.gro', p=top, o=outStr)
            # water protein
            if bWatProt==True:            
                inStr = '{0}/box.pdb'.format(outProtPath)
                outStr = '{0}/water.pdb'.format(outProtPath)
                top = '{0}/topolB.top'.format(outProtPath)
                gmx.solvate(inStr, cs='spc216.gro', p=top, o=outStr)
            
            # ions ligand
            if bIonLig:
                inStr = '{0}/water.pdb'.format(outLigPath)
                outStr = '{0}/ions.pdb'.format(outLigPath)
                mdp = '{0}/em_l0.mdp'.format(self.mdpPath)
                tpr = '{0}/tpr.tpr'.format(outLigPath)
                top = '{0}/topol.top'.format(outLigPath)
                mdout = '{0}/mdout.mdp'.format(outLigPath)
                gmx.grompp(f=mdp, c=inStr, p=top, o=tpr, maxwarn=4, other_flags=' -po {0} > {1}/genion.log 2>&1'.format(mdout, outLigPath))        
                gmx.genion(s=tpr, p=top, o=outStr, conc=self.conc, neutral=True, 
                      other_flags=' -pname {0} -nname {1} >> {2}/genion.log 2>&1'.format(self.pname, self.nname, outLigPath))
            # ions protein
            if bIonProt:
                inStr = '{0}/water.pdb'.format(outProtPath)
                outStr = '{0}/ions.pdb'.format(outProtPath)
                mdp = '{0}/em_l0.mdp'.format(self.mdpPath)
                tpr = '{0}/tpr.tpr'.format(outProtPath)
                top = '{0}/topolB.top'.format(outProtPath)
                mdout = '{0}/mdout.mdp'.format(outProtPath)
                index = '{0}/genion.ndx'.format(outProtPath)
                
                #create an index file
                os.system(f"echo 'q\n' | gmx make_ndx -f {inStr} -o {index} > {outProtPath}/genion.log 2>&1")
                #clean it and separate SOL into crystal and non-crystal groups
                ndx_file = ndx.IndexFile(index)

                xtal_sol_ids=[ i for i in ndx_file["SOL"].ids if i<ndx_file["UNL"].ids[0] ]
                g = ndx.IndexGroup(ids=xtal_sol_ids, name="xtal_SOL")
                ndx_file.add_group(g)

                sol_ids=[ i for i in ndx_file["SOL"].ids if i>ndx_file["UNL"].ids[0] ]
                g = ndx.IndexGroup(ids=sol_ids, name="SOL")
                ndx_file.delete_group("SOL")
                ndx_file.add_group(g)

                ndx_file.write(index)
                
                #make the ions
                gmx.grompp(f=mdp, c=inStr, p=top, o=tpr, maxwarn=4, other_flags=' -po {0} -n {1}  >> {2}/genion.log 2>&1'.format(mdout, index, outProtPath))        
                gmx.genion(s=tpr, p=top, o=outStr, conc=self.conc, neutral=True, 
                      other_flags=' -pname {0} -nname {1} -n {2}  >> {3}/genion.log 2>&1'.format(self.pname, self.nname, index, outProtPath))
                
                #copy water & ions from topolA to topolB
                os.system(f"sed -n '/MOL /,$p' {outProtPath}/topolA.top | tail -n +2 >> {outProtPath}/topolB.top")
                
            # clean backed files
            self._clean_backup_files( outLigPath )
            self._clean_backup_files( outProtPath )
        
        if edges==None:
            edges = self.edges
            
        if(n_processes==1):
            for edge in edges:
                end_state_boxWaterIons_Single((edge, bBoxLig, bBoxProt, bWatLig, bWatProt, bIonLig, bIonProt))
        elif(n_processes>1):
            args=[(edge, bBoxLig, bBoxProt, bWatLig, bWatProt, bIonLig, bIonProt) for edge in edges]
            with Pool(processes=n_processes) as pool:
                res=my_pool_map(pool, end_state_boxWaterIons_Single, args)
        else:
            raise(Exception("Invalid number of prcocesses requested:", n_processes))

        print('DONE')
