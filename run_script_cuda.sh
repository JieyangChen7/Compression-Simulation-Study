#!/bin/bash
#BSUB -P CSC331
#BSUB -W 01:00
#BSUB -nnodes 64
#BSUB -J test
#BSUB -o test.%J
#BSUB -e test.%J

ml cmake cuda/11.4 adios2 libfabric gcc/9

set -e
# set -x
USE_COMPRESSION=1
VERBOSE=0
NUM_GPU=1
REORDER=0
LOSSLESS=0
SIM_ITER=1
ACCUMULATE_DATA=10
COMPUTE_DELAY=0
PREFETCH=1
DEVICE=cuda

./build_script_cuda.sh

#JSRUN='jsrun -n 6 -a 1 -c 1 -g 1 -r 6 -l gpu-cpu --smpiargs="-disable_gpu_hooks"'
JSRUN='jsrun -n 384 -a 1 -c 1 -g 1 -r 6 -l gpu-cpu'

NSYS='nsys profile -o /gpfs/alpine/csc143/proj-shared/jieyang/nsys_%q{OMPI_COMM_WORLD_RANK} --force-overwrite true'
SIM=./build_cuda/cpu-application-simulator

DATA=$HOME/dev/data/d3d_coarse_v2_700.bin


set -x

EB=1e17
ACCUMULATE_DATA=10
MAX_MEM=16e9
VERBOSE=0
rm -rf $DATA.bp
$JSRUN $SIM -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/${DATA}_cuda.bp \
            -t d -n 3 312 1093 585 -m abs -e $EB -s inf -r $REORDER \
            -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
            -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -d $DEVICE

#done
exit 0


#rm -rf $DATA.bp
#$JSRUN $SIM -z $USE_COMPRESSION \
#            -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/$DATA.bp \
#            -t d -n 4 8 39 16395 39 -m abs -e $EB -s 0 -r $REORDER \
#            -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
#            -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -h $PREFETCH -d $DEVICE

#rm -rf $DATA.bp
#$JSRUN $SIM -z $USE_COMPRESSION \
#            -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/$DATA.bp \
#            -t d -n 3 312 16395 39 -m abs -e $EB -s 0 -r $REORDER \
#            -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
#            -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -h $PREFETCH -d $DEVICE

#rm -rf $DATA.bp
#$JSRUN $SIM -z $USE_COMPRESSION \
#            -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/$DATA.bp \
#            -t d -n 3 312 1093 585 -m abs -e $EB -s 0 -r $REORDER \
#            -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
#            -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -h $PREFETCH -d $DEVICE

if ((0))
then

for ACCUMULATE_DATA in 1 2 4 6 8 10
do
  EB=1e17
  PREFETCH=1
  USE_COMPRESSION=0
  rm -rf $DATA.bp
  $JSRUN $SIM -z $USE_COMPRESSION \
              -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/$DATA.bp \
              -t d -n 3 312 1093 585 -m abs -e $EB -s inf -r $REORDER \
              -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
              -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -h $PREFETCH -d $DEVICE
done

fi

for ACCUMULATE_DATA in 1 2 4 6 8 10
do
  EB=1e17
  PREFETCH=0
  USE_COMPRESSION=1
  rm -rf $DATA.bp 
  $JSRUN $SIM -z $USE_COMPRESSION \
              -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/$DATA.bp \
              -t d -n 3 312 1093 585 -m abs -e $EB -s inf -r $REORDER \
              -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
              -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -h $PREFETCH -d $DEVICE
done



for ACCUMULATE_DATA in 1 2 4 6 8 10
do
  EB=1e17
  PREFETCH=1
  USE_COMPRESSION=1
  rm -rf $DATA.bp
  $JSRUN $SIM -z $USE_COMPRESSION \
              -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/$DATA.bp \
              -t d -n 3 312 1093 585 -m abs -e $EB -s inf -r $REORDER \
              -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
              -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -h $PREFETCH -d $DEVICE
done

#fi

for EB in 1e13 1e14 1e15 1e16 1e17 1e18
do 
  ACCUMULATE_DATA=10
  PREFETCH=1
  USE_COMPRESSION=1
  rm -rf $DATA.bp
  $JSRUN $SIM -z $USE_COMPRESSION \
              -i $DATA -c /gpfs/alpine/csc143/proj-shared/jieyang/$DATA.bp \
              -t d -n 3 312 1093 585 -m abs -e $EB -s inf -r $REORDER \
              -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER \
              -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -h $PREFETCH -d $DEVICE
done


