#!/bin/bash 
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/mylog
#PBS -N MOE_DP_before_EP


##PBS -l min_walltime=01:00:00
module reset
module load frameworks/2025.0.0
source /lus/flare/projects/Aurora_deployment/eku/venv/base/bin/activate

export PBS_O_WORKDIR=$(dirname $0 | xargs realpath)
export DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt
export OPT=adamw
export GRAD_ACC_STEPS=1
export TRAIN_ITERS=5
export EVAL_ITERS=1
export LOG_INTERVAL=1
export WANDB_DISABLED=1
export FLOPS_PROFILER=true
export COMMS_LOGGER=true

## Env Variables
vaino_env_var(){
    cd "${PBS_O_WORKDIR}" || exit
    HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
    GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH

    export CCL_KVS_MODE=mpi
    export CCL_KVS_CONNECTION_TIMEOUT=600 
    export PALS_PMI=pmix # Required by Aurora mpich
    export CCL_ATL_TRANSPORT=mpi # Required by Aurora mpich

    export CCL_OP_SYNC=1
    export CCL_ENABLE_AUTO_CACHE=0
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=4096

    export FI_CXI_DEFAULT_CQ_SIZE=1048576
    export FI_CXI_RX_MATCH_MODE=hybrid
    export FI_MR_CACHE_MONITOR=disabled
    export FI_CXI_OFLOW_BUF_SIZE=8388608
    export FI_CXI_CQ_FILL_PERCENT=30

    export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
    export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
    export NUMEXPR_MAX_THREADS=7
    export OMP_NUM_THREADS=7

    export PALS_PING_PERIOD=480
    export PALS_RPC_TIMEOUT=480

    # export DATA_FILE_LIST="/flare/Aurora_deployment/vhat/AuroraGPT_2025.0.0/Megatron-DeepSpeed/filelist_debug.txt"
}

vaino_env_var

## Model
export NLAYERS=24
export HIDDEN=4680
export FFN_HIDDEN_SIZE=$((HIDDEN * 12 * 3))
export HEADS=36
export NUM_KV_HEAD=12
export SEQ=4104
export SP=1
export PP=1
export TP=12
export MICRO_BATCH=12
export ZERO_STAGE=1
export USE_ACTIVATION_CHECKPOINTING=1


## RUN
cd "${PBS_O_WORKDIR}" || exit
cd ..  # Since we are inside myscripts directory
HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH
# 2. source `ALCF/helpers.sh`
source "${HERE}/ALCF/helpers.sh" || exit
# 3. call `setup` from `./ALCF/helpers.sh`
setup "$@ $extra_ds_args" || exit
echo "${run_cmd[@]}" | tee -a "${OUTPUT_LOG}"
# 4. Tell user where to find output
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
# 5. run cmd
eval "${run_cmd[*]}" |& tee "/lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/train.log"