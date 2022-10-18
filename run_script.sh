
set -e
# set -x
USE_COMPRESSION=0
VERBOSE=0
NUM_GPU=1
REORDER=0
LOSSLESS=0
SIM_ITER=1
ACCUMULATE_DATA=2
COMPUTE_DELAY=0

SIM=./build/cpu-application-simulator


DATA=$HOME/dev/data/d3d_coarse_v2_700.bin
rm -rf $DATA.bp
mpirun -np 1 $SIM -z $USE_COMPRESSION -i $DATA -c $DATA.bp -t d -n 3 312 1093 585 -m abs -e 1e17 -s 0 -r $REORDER -l $LOSSLESS -g $NUM_GPU -v $VERBOSE -p $SIM_ITER -a $ACCUMULATE_DATA -k $COMPUTE_DELAY -d $1