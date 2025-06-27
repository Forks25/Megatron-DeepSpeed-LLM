#!/bin/bash 
#PBS -l select=8
#PBS -l min_walltime=01:00:00
#PBS -l walltime=01:00:00
#PBS -q prod
#PBS -A AuroraGPT
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/my_log
#PBS -N MOE_DP_before_EP


export NO_COLOR=1
# TZ="America/Chicago" date
# DATE=$()
# echo $DATE
TZ='America/Chicago' date +%F_%H%M

# ------------------------------------------------------------------------------------ #

# 1. Set-up modules and env
module reset
module load frameworks  # /2025.0.0
source /lus/flare/projects/Aurora_deployment/eku/venv/base/bin/activate

# 2. Env Variables
aurora_env_var(){
    export CCL_KVS_MODE=mpi
    export CCL_KVS_CONNECTION_TIMEOUT=600 
    export PALS_PMI=pmix # Required by Aurora mpich
    export CCL_ATL_TRANSPORT=mpi # Required by Aurora mpich

    export CCL_OP_SYNC=1
    export CCL_ENABLE_AUTO_CACHE=0
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=4096

    export FI_CXI_DEFAULT_CQ_SIZE=1048576
    export FI_CXI_RX_MATCH_MODE=hybrid
    export FI_MR_CACHE_MONITOR=kdreg2 #disabled
    export FI_CXI_OFLOW_BUF_SIZE=8388608
    export FI_CXI_CQ_FILL_PERCENT=30

    export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
    export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
    export NUMEXPR_MAX_THREADS=7
    export OMP_NUM_THREADS=7

    export PALS_PING_PERIOD=480
    export PALS_RPC_TIMEOUT=480
}
aurora_env_var  # set-up env variables

# for perf reasons
# export CCL_ALLGATHERV=topo
# export CCL_ALLREDUCE=topo
export CCL_BCAST=topo  # prevent possible hang?

# ------------------------------------------------------------------------------------ #

# Get repo root dir
# BASH_SOURCE or $0 is relative if the invokcation was also relative. This can give you incorrect result if you cd in between invokation and dirname.
# cd $PBS_O_WORKDIR
# echo $(pwd)
# exit
## TODO: Fix PBS_O_WORKDIR
# if [[ -z $PBS_O_WORKDIR ]]; then
#     echo got here
#     cd $(dirname ${BASH_SOURCE[0]})
#     cd ..
#     export PBS_O_WORKDIR=$(pwd)
# fi

# ------------------------------------------------------------------------------------ #

# 1. Set-up train config
export DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt
export OPT=adamw
export GRAD_ACC_STEPS=${GAS:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-5}
export EVAL_ITERS=1
export LOG_INTERVAL=1
export FLOPS_PROFILER=true
export COMMS_LOGGER=true
# export TORCH_PROFILER_ENABLE=2
# export LOG_LEVEL="WARNING"
# export EZPZ_LOG_LEVEL="WARNING"

# 2. Set-up model config
# sam's 70B cfg 
# export HIDDEN=8192; export HEADS=64; export NUM_KV_HEAD=8; export SEQ=8192; export FFN_HIDDEN_SIZE=28672  # Usually 3*Hidden for GLU multiplier
# divisible cfg 
export HIDDEN=8160; export HEADS=60; export NUM_KV_HEAD=12; export SEQ=8160; export FFN_HIDDEN_SIZE=28560  # Usually 3*Hidden for GLU multiplier
export NLAYERS=${NLAYERS:-1}  # 80 for 69B

export SP=${SP:-1}
export PP=1
export TP=${TP:-1}
export MICRO_BATCH=${MBS:-1}
export ZERO_STAGE=${ZERO_STAGE:-3}
export USE_ACTIVATION_CHECKPOINTING=1
# if [[ $MICS_SHARD_SIZE -gt 1 ]]; then
#     # Custom DeepSpeed with MICS fix
# fi

# 3. Set-up MoE config
# export PYTHONPATH="/lus/flare/projects/Aurora_deployment/eku/tests/test_MICS/MDS-MICS/deps:${PYTHONPATH}"
## MoE variables
# num_experts=12
# extra_ds_args="--num-experts $num_experts --expert-interval 1 --create-moe-param-group --topk 2"
# extra_ds_args="$extra_ds_args --moe-expert-parallel-size $num_experts"
# extra_ds_args="$extra_ds_args --enable-expert-tensor-parallelism"

# ------------------------------------------------------------------------------------ #

## RUN
cd "${PBS_O_WORKDIR}" || exit
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
log_pth="/lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/train.log"
eval "${run_cmd[*]}" |& tee $log_pth