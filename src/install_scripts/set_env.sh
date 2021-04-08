#!/bin/sh
export MORPHOGENESIS_VERSION=0.1.1
if [[ $module_present == "installed" ]]; then
mkdir -p ~/modules/morphogenesis
cp modules/morphogenesis/* ~/modules/morphogenesis
module use ~/modules
module load morphogenesis
else
export  MORPHOGENESIS_HOME=~/apps/morphogenesis/$MORPHOGENESIS_VERSION
export  LD_RUN_PATH=$MORPHOGENESIS_HOME/lib64:$LD_RUN_PATH
export  LD_LIBRARY_PATH=$MORPHOGENESIS_HOME/lib64:$LD_LIBRARY_PATH
export  PATH=$MORPHOGENESIS_HOME/bin:$PATH
fi


