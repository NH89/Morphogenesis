#!/bin/sh

return=0

command_exists() {
    # check if command exists and fail otherwise
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


# check prerequisites
#echo "CUDACXX=$CUDACXX"
command_exists "$CUDACXX"
CUDACXX_present=$return

command_exists "nvcc"
nvcc_present=$return

if [[ ! ( $CUDACXX_present == "installed" || $nvcc_present == "installed" ) ]]
then 
    echo "Please set CUDA environment variables before building"
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
    #echo $ccmake_version
    [[ "$ccmake_version" =~ ([0-9]*)([^0-9]*)([0-9]*)(.*)  ]]
    #echo "BASH_REMATCH[0] = " ${BASH_REMATCH[0]}
    #echo "BASH_REMATCH[1] = " ${BASH_REMATCH[1]}
    #echo "BASH_REMATCH[3] = " ${BASH_REMATCH[3]}
    if [[  ! ( ${BASH_REMATCH[1]} -ge 3  && ${BASH_REMATCH[3]} -ge 8 )  ]]
    then
        echo "Ccmake version ${BASH_REMATCH[0]} insufficient, please install ccmake 3.8 or above before building"
        exit 1
    else
        echo "ccmake found"
    fi
fi
