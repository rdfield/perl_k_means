#!/bin/bash
export MLDIR=`pwd`
cd $MLDIR/amd_kernel
cmake -S . -B build
cmake --build build
cmake --install build
cd $MLDIR/nvidia_kernel
cmake -S . -B build
cmake --build build
cmake --install build
cd $MLDIR

