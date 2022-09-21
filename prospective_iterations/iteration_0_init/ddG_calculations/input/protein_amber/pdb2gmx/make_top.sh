#!/bin/bash

export GMXLIB=/home/ykhalak/private/Vytas_old_gmx_share_top

rm *.top .
rm *.itp .

#gmx 2019.3
#echo -e "5\n1\n1\n0\n0\n1\n0\n1\n1\n0\n1\n1\n1\n1\n1\n2\n1\n1\n1" | gmx pdb2gmx -f ../4d08_chain_A_matched.pdb -ignh -o out.pdb -his
echo -e "10\n1\n1\n0\n0\n1\n0\n1\n1\n0\n1\n1\n1\n1\n1\n2\n1\n1\n1" | gmx pdb2gmx -f 4d09_aligned_w_water.pdb -ignh -o out.pdb -his

rm \#*

