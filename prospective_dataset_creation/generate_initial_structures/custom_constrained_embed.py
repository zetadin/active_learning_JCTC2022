from rdkit import rdBase
from rdkit import RDConfig
from rdkit import DataStructs
from rdkit.Geometry import rdGeometry
from rdkit.Chem import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.Chem.rdChemReactions import *
try:
  from rdkit.Chem.rdSLNParse import *
except:
  pass
from rdkit.Chem.rdMolDescriptors import *
from rdkit import ForceField
import numpy

from rdkit.Chem.AllChem import *

def CustomConstrainedEmbed(mol,core,coordMap,algMap,useTethers=True,coreConfId=-1,
                     randomseed=2342, debug=False):
    """ generates an embedding of a molecule where part of the molecule
    is constrained to have particular coordinates
    Arguments
        - mol: the molecule to embed
        - core: the molecule to use as a source of constraints
        - useTethers: (optional) if True, the final conformation will be
            optimized subject to a series of extra forces that pull the
            matching atoms to the positions of the core atoms. Otherwise
            simple distance constraints based on the core atoms will be
            used in the optimization.
        - coreConfId: (optional) id of the core conformation to use
        - randomSeed: (optional) seed for the random number generator
        - coordMap:   (optional) dict of coordinates with constrained mol atom indeces as keys
    An example, start by generating a template with a 3D structure:
    >>> from rdkit.Chem import AllChem
    >>> template = AllChem.MolFromSmiles("c1nn(Cc2ccccc2)cc1")
    >>> AllChem.EmbedMolecule(template)
    0
    >>> AllChem.UFFOptimizeMolecule(template)
    0
    Here's a molecule:
    >>> mol = AllChem.MolFromSmiles("c1nn(Cc2ccccc2)cc1-c3ccccc3")
    Now do the constrained embedding
    >>> newmol=AllChem.ConstrainedEmbed(mol, template)
    Demonstrate that the positions are the same:
    >>> newp=newmol.GetConformer().GetAtomPosition(0)
    >>> molp=mol.GetConformer().GetAtomPosition(0)
    >>> list(newp-molp)==[0.0,0.0,0.0]
    True
    >>> newp=newmol.GetConformer().GetAtomPosition(1)
    >>> molp=mol.GetConformer().GetAtomPosition(1)
    >>> list(newp-molp)==[0.0,0.0,0.0]
    True
    """
  

    match = list(coordMap.keys())
        
    if(debug):
        print(coordMap)

    ci = EmbedMolecule(mol,coordMap=coordMap,randomSeed=randomseed)
    if ci<0:
        raise(ValueError('Could not embed molecule.'))
    
    if(debug):
        print("embeding returned", ci)
  
    algMap=[(j,i) for i,j in enumerate(coordMap.keys())]
    
    if(debug):
        print("algMap:", algMap)
        print("useTethers:", useTethers)

    if not useTethers:
        # clean up the conformation
        ff = UFFGetMoleculeForceField(mol,confId=0)
        for i,idxI in enumerate(match):
            for j in range(i+1,len(match)):
                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI,idxJ,d,d,100.)
        ff.Initialize()
        n=4
        more=ff.Minimize()
        while more and n:
              more=ff.Minimize()
              n-=1
        # rotate the embedded conformation onto the core:
        rms =AlignMol(mol,core,atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        rms = AlignMol(mol,core,atomMap=algMap)
        ff =  UFFGetMoleculeForceField(mol,confId=0)
        conf = core.GetConformer()
        for i in range(len(match)):
            #p =conf.GetAtomPosition(i)
            p = coordMap[match[i]]
            pIdx=ff.AddExtraPoint(p.x,p.y,p.z,fixed=True)-1
            ff.AddDistanceConstraint(pIdx,match[i],0,0,100.)
        ff.Initialize()
        n=4
        more=ff.Minimize(energyTol=1e-4,forceTol=1e-3)
        while more and n:
            more=ff.Minimize(energyTol=1e-4,forceTol=1e-3)
            n-=1
        # realign
        rms = AlignMol(mol,core,atomMap=algMap)
    mol.SetProp('EmbedRMS',str(rms))
    return mol


