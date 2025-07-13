#!/bin/bash -l
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/my_log
#PBS -N LLMPerf

# ------------------------------------------------------------------------------------ #

# module load pti-gpu
# module load thapi
export PBS_O_WORKDIR="/lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed"
# export THAPI_HOME=PBS_O_WORKDIR

cd /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/my_scripts/MICS
num_nodes=$(wc -l < $PBS_NODEFILE)
suffix=""
run_exp() {
    exp_name="n${num_nodes}_l${NLAYERS}${suffix}"
    mkdir -p $exp_name || exit

    # # 1. normal
    bash dense70B.sh |& tee $exp_name/vanilla_Z3.log

    # # 2. MICS
    # MICS_SHARD_SIZE=$MICS_SHARD_SIZE bash dense70B.sh |& tee $exp_name/MICS.log

    # 3. HPZ
    # HPZ_SHARD_SIZE=$HPZ_SHARD_SIZE bash dense70B.sh |& tee $exp_name/HPZ.log
}

export TRAIN_ITERS=10000
export NLAYERS=20
export GAS=8
# export TORCH_PROFILER_ENABLE=2
# MICS_SHARD_SIZE=12
# HPZ_SHARD_SIZE=12


# reduce_bucket_size=90000000 suffix="${suffix}_rbs9e7" run_exp
reduce_bucket_size=1024000000 suffix="${suffix}_rbs1e9" run_exp
# reduce_bucket_size=2048000000 suffix="${suffix}_rbs2e9" run_exp
# reduce_bucket_size=4096000000 suffix="${suffix}_rbs4e9" run_exp

# export SP=12; export MBS=6; suffix="${suffix}_SP${SP}_MBS${MBS}"