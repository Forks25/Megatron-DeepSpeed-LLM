#!/bin/bash 
#PBS -l select=8
#PBS -l min_walltime=01:00:00
#PBS -l walltime=01:00:00
#PBS -q prod
#PBS -A AuroraGPT
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/mylog
#PBS -N MOE_DP_before_EP


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
}
vaino_env_var  # set-up env variables

# ------------------------------------------------------------------------------------ #

module reset
module load frameworks/2025.0.0
source /lus/flare/projects/Aurora_deployment/eku/venv/base/bin/activate

# Get repo root dir
PBS_O_WORKDIR=$(dirname $0)
# cd $PBS_O_WORKDIR
# echo $(pwd)
# exit
## TODO: Fix PBS_O_WORKDIR
# if [[ -z $PBS_O_WORKDIR ]]; then
#     echo got here
#     cd $(dirname $0)
#     cd ..
#     export PBS_O_WORKDIR=$(pwd)
# fi
# echo $PBS_O_WORKDIR


export DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt
export OPT=adamw
export GRAD_ACC_STEPS=1
export TRAIN_ITERS=5
export EVAL_ITERS=1
export LOG_INTERVAL=1
export WANDB_DISABLED=1
# export FLOPS_PROFILER=true
# export COMMS_LOGGER=true
# export TORCH_PROFILER_ENABLE=2  # Enable torch profiler

## Model
export NLAYERS=1
export HIDDEN=8160
export FFN_HIDDEN_SIZE=$((3*HIDDEN))
export HEADS=60
export NUM_KV_HEAD=12
export SEQ=8160
export SP=1
export PP=4
export TP=1
export GRAD_ACC_STEPS=$((2*PP))
export MICRO_BATCH=1
export ZERO_STAGE=1
# export USE_ACTIVATION_CHECKPOINTING=1

## MoE variables
num_experts=3
extra_ds_args="--num-experts $num_experts --expert-interval 1 --create-moe-param-group --topk 2"
extra_ds_args="$extra_ds_args --moe-expert-parallel-size $num_experts"
# extra_ds_args="$extra_ds_args --enable-expert-tensor-parallelism"

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