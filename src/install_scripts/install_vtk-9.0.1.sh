#!/bin/sh
echo "\n NB must use the same c++11 library and compiler as Morphogenesis, to ensure ABI compastability."

VTK_VERSION=9.0.1
mkdir -p ~/apps/vtk/$VTK_VERSION

if (command_exists(module))
mkdir -p ~/modules/vtk
cp modules/vtk/* ~/modules/vtk
module use ~/modules
module load vtk/$VTK_VERSION

else #(modules not installed)
export  VTK_HOME=~/apps/vtk/$VTK_VERSION
export  LD_RUN_PATH=$VTK_HOME/lib64:$LD_RUN_PATH
export  LD_LIBRARY_PATH=$VTK_HOME/lib64:$LD_LIBRARY_PATH
export  PATH=$VTK_HOME/bin:$PATH
endif

mkdir -p ../vtk-$VTK_VERSION/build
cd ../vtk-$VTK_VERSION
git@gitlab.kitware.com:vtk/vtk.git
cd vtk
git checkout v9.0.1
cd ../build
ccmake ../vtk

make install -D CMAKE_INSTALL_PREFIX $VTK_HOME
