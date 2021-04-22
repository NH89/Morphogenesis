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


echo 'Checking prerequisites'
command_exists "$CUDACXX"
CUDACXX_present=$return

command_exists "nvcc"
nvcc_present=$return

if [[ ! ( $CUDACXX_present == "installed" || $nvcc_present == "installed" ) ]]
then 
    echo "Please set CUDA environment variables before building, e.g. export CUDACXX=<path_to_nvcc>"
    exit 1
else
    echo "CUDA compiler found"
fi

command_exists "ccmake"
ccmake_present=$return
echo "$ccmake_present=$ccmake_present"

if [[ ! ( $ccmake_present == "installed" ) ]]
then
    echo "Please install ccmake version 3.8 or greater before building."
    exit 1
else
    ccmake_version=$(cmake --version | head -n1 | cut -d" " -f3)
    [[ "$ccmake_version" =~ ([0-9]*)([^0-9]*)([0-9]*)(.*)  ]]
    if [[  ! ( ${BASH_REMATCH[1]} -ge 3  && ${BASH_REMATCH[3]} -ge 8 )  ]]
    then
        echo "ccmake version ${BASH_REMATCH[0]} insufficient, please install ccmake 3.8 or above before building"
        exit 1
    else
        echo "ccmake found"
    fi
fi


echo 'Setting environment variables'
export MORPHOGENESIS_VERSION=0.1.1
mkdir -p ~/apps/morphogenesis/$MORPHOGENESIS_VERSION

command_exists "module"
module_present=$return

if [[ $module_present == "installed" ]]; then
mkdir -p ~/modules/morphogenesis
cp modules/morphogenesis/* ~/modules/morphogenesis
module use ~/modules
module load morphogenesis
module avail morphogenesis
module list
env | grep morphogenesis
echo $PATH
else
export  MORPHOGENESIS_HOME=~/apps/morphogenesis/$MORPHOGENESIS_VERSION
export  LD_RUN_PATH=$MORPHOGENESIS_HOME/lib64:$LD_RUN_PATH
export  LD_LIBRARY_PATH=$MORPHOGENESIS_HOME/lib64:$LD_LIBRARY_PATH
export  PATH=$MORPHOGENESIS_HOME/bin:$PATH
fi

echo 'Creating directories, & moving old directories'
NOW=`date +"%Y-%b-%d-%H:%M"`
echo "working directory = $PWD"
if [[ -e ../data ]] 
then
    echo "moving ../data to ../old-data$NOW"
    mv ../data "../old-data$NOW" 
fi
if [[ -e ../build ]]
then
    echo "moving ../build to ../old-build$NOW"
    mv ../build "../old-build$NOW"
fi

mkdir -p ../data/demo
mkdir -p ../data/check
mkdir -p ../data/out
mkdir -p ../data/test

mkdir -p ../build
cd ../build

echo 'Building & installing Morphogenesis'
ccmake ../src
make install

module list 
module avail morphogenesis
env | grep morphogenesis
echo $PATH

echo 'Generating Morphogenesis Demo model'
cd ../data
make_demo   #use default


echo 'Checking read-write of model'
check_demo  demo  check

#echo 'Test run of simulation'
#make_demo2  demo


echo 'Install of Morphogenesis complete'
