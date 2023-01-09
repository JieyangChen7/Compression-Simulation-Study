#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building the example

set -x
set -e

mgard_src_dir=/ccs/home/jieyang/MGARD
mgard_install_dir=${mgard_src_dir}/install-hip-crusher

export LD_LIBRARY_PATH=${mgard_install_dir}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${mgard_install_dir}/lib64:${LD_LIBRARY_PATH}


rm -rf build_hip
mkdir -p build_hip 
cmake -S .  -B ./build_hip \
	    -DCMAKE_MODULE_PATH=${mgard_src_dir}/cmake\
      -DCMAKE_C_COMPILER=cc\
      -DCMAKE_CXX_COMPILER=CC\
	    -Dmgard_ROOT=${mgard_src_dir}/install-hip-crusher\
	    -DCMAKE_PREFIX_PATH="${mgard_install_dir}"

#cd build_hip && make && cd ..

cmake --build ./build_hip
