#!/bin/bash 
#PBS -l select=2
#PBS -l min_walltime=01:00:00
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/my_log
#PBS -N MOE_DP_before_EP

# MoE Experiment
# script_pth="/lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/my_scripts/moe_functionality/moe_EP_aurora.sh"
# cd /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed

# export NEXPERTS=2
# bash $script_pth |& tee train3.log

# export NEXPERTS=3
# bash $script_pth |& tee train4.log

# export NEXPERTS=4
# bash $script_pth |& tee train5.log

# export NEXPERTS=6
# bash $script_pth |& tee train7.log

# export NEXPERTS=12
# bash $script_pth |& tee train8.log


cd "/flare/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed"
script_pth="my_scripts/grad-sync/dense.sh"

# ZERO_STAGE=0 bash $script_pth |& tee train1.log
# ZERO_STAGE=2 bash $script_pth |& tee train2.log
ZERO_STAGE=3 bash $script_pth |& tee train3.log

module restore
module load frameworks
source /lus/flare/projects/Aurora_deployment/eku/venv/base/bin/activate
pip show torch
# ZERO_STAGE=0 bash $script_pth |& tee train4.log
# ZERO_STAGE=2 bash $script_pth |& tee train5.log
ZERO_STAGE=3 bash $script_pth |& tee train6.log