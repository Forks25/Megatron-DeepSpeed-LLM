module reset
module load frameworks/2025.0.0
# source /lus/flare/projects/Aurora_deployment/eku/scaling_MDS/Sams_Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2025.0.0/bin/activate
source /lus/flare/projects/Aurora_deployment/eku/venv/base/bin/activate


## TIP: PBS_O_WORKDIR is automatically set to the parent directory of the script when submitted through qsub. 
export PBS_O_WORKDIR=$(dirname $0 | xargs realpath)
export DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt
export OPT=adamw
export GRAD_ACC_STEPS=1
export TRAIN_ITERS=10
export EVAL_ITERS=1

## Model
export NLAYERS=24
export HIDDEN=2400
export FFN_HIDDEN_SIZE=2400
export HEADS=12
export NUM_KV_HEAD=12
export SEQ=4104
export SP=1
export PP=1
export TP=1
export ZERO_STAGE=0
export MICRO_BATCH=1


quick_check="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --disable-bias-linear --normalization rmsnorm --attention-dropout 0 --hidden-dropout 0 --num-key-value-heads 8 --use-flash-attn-builder --num-experts 12 --create-moe-param-group --moe-expert-parallel-size 12 --topk 2 --expert-interval 1"

bash $PBS_O_WORKDIR/train_aGPT_7B.sh $quick_check |& tee $PBS_O_WORKDIR/train.log