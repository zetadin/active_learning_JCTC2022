#!/bin/bash

fold="/home/ykhalak/Projects/ML_dG/pde2_dG/generators/structures/set_3/morphing_annealing/data/setup_energ/"
cat index.ndx $fold/close_prot_residues.ndx > index_for_energ.ndx

for i in 1 2 3
do
    if [ -s "energy_block_$i.xvg" ]
    then 
        continue
    else
        gmx grompp -f $fold/lig_prot_energ_block$i.mdp -c morph.gro -o temp_lig_prot_energ_block$i.tpr -r combined.pdb  -n index_for_energ.ndx -maxwarn 3 >/dev/null 2>&1
        mdrun_threads -ntomp 4 -deffnm temp_lig_prot_energ_block$i -rerun morph.gro >/dev/null 2>&1
        gmx energy -f temp_lig_prot_energ_block$i.edr -o energy_block_$i.xvg -xvg none < $fold/g_energy_block_${i}_input.txt >/dev/null 2>&1
        
        rm temp_lig_prot_energ_block$i.*
    fi   
done

