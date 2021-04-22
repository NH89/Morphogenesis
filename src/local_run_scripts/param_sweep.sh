#!/bin/bash

FOLDER=($HOME/Programming/Cuda/Morphogenesis/data/demo_batch/demo_intstiff*.000000/)

while(${FOLDER[ITERATOR]} != NULL)
do
echo ${FOLDER[ITERATOR]}
make_demo2 ${FOLDER[ITERATOR]} ${FOLDER[ITERATOR]}
ITERATOR++
done

# bad syntax, correct to make valid bash script.
