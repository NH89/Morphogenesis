#!/bin/bash

# Script to rename and collate vtk files, from parameter sweep, so that they can be viewedas series in Paraview.
# NB make sure that the only .vtk files are the original param sweep output, in their original directories.

for i in $(ls)
do
    cd $i
    for j in $(ls *.vtp)
    do
        echo $i.vtp
        cp $j ../$i.vtp
    done
    cd ../
done
