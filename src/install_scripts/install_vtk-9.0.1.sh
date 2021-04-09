#!/bin/sh

return=0

# check if command exists and fail otherwise
command_exists() {
    if [[ $# -eq 0 ]]
    then 
        echo "string empty" 
        return="empty" #exit 1
    fi
    command -v "$1" >/dev/null 2>&1
    if [[ $? -ne 0 ]]
    then
        echo "command $1 not installed" 
        return="missing" #exit 1
    else
        echo "command $1 installed"
        return="installed"
    fi
    #echo $return
}


echo "\n NB must use the same c++11 library and compiler as Morphogenesis, to ensure ABI compastability."

VTK_VERSION=9.0.1
mkdir -p ~/apps/vtk/$VTK_VERSION

command_exists "module"
module_present=$return

if [[ $module_present == "installed" ]]; 
    then
    mkdir -p ~/modules/vtk
    cp modules/vtk/* ~/modules/vtk
    module use ~/modules
    module load vtk/$VTK_VERSION
else #(modules not installed)
    export  VTK_HOME=~/apps/vtk/$VTK_VERSION
    export  LD_RUN_PATH=$VTK_HOME/lib64:$LD_RUN_PATH
    export  LD_LIBRARY_PATH=$VTK_HOME/lib64:$LD_LIBRARY_PATH
    export  PATH=$VTK_HOME/bin:$PATH
fi

mkdir -p ../vtk-$VTK_VERSION/build
cd ../vtk-$VTK_VERSION
wget https://www.vtk.org/files/release/9.0/VTK-9.0.1.tar.gz
tar -xvf VTK-9.0.1.tar.gz
cd build
ccmake -D CMAKE_INSTALL_PREFIX=$VTK_HOME ../VTK-9.0.1

make install 



#git clone git@gitlab.kitware.com:vtk/vtk.git
#git checkout v9.0.1
#cd ../build

