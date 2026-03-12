set -e
set -x
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset HSA_VISIBLE_DEVICES
VISIBLE_DEVICES="0,1"
export HYDRA_FULL_ERROR=1


ROOT_DIR=$(pwd)
TRAIN_FILE=$ROOT_DIR/data/deepscaler/train.parquet
GSM8K_VAL_PATH=$ROOT_DIR/data/gsm8k/test.parquet

VAL_PREFIX=$ROOT_DIR/data/benchmarks
MATH500_PATH=$VAL_PREFIX/math500.parquet
VAL_FILE_LIST="['$AIME25_PATH','$MATH500_PATH']"


LR=1e-6
BACKBONE="R1-Distill-Qwen-1.5B"
BACKBONE_PATH="your_model_path"
MAX_PROMPT_LENGTH=1024
MAX_GEN_LENGTH=3500
MODEL_ID="R1-Distill-Qwen-1.5B"
DATE=$(date +"%m%d_%H%M")
TASK="OPSFT"
DATASET_NAME="dsr"
ROLLOUT_N=32
EXPERIMENT="CE-${DATASET_NAME}"
ENABLE_TRAIN_TEMP=False


PROJECT_NAME="On_Policy_SFT_deepscaler_r1_distill_1.5b"

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs

MODEL="${TASK}-${BACKBONE}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${MODEL}/${TASK}/${EXPERIMENT}/${DATE}"

mkdir -p ${OUTPUT_DIR}

EXP="${TASK}-${MODEL_ID}-${EXPERIMENT}-lr${LR}-TAUS${TAU_S}-rollout${ROLLOUT_N}-ttr${ENABLE_TRAIN_TEMP}-${DATE}"

export SWANLAB_API_KEY="your_swanlab_api_key"
export SWANLAB_LOG_DIR=${ROOT_DIR}/logs/swanlab/${EXP}

mkdir -p ${SWANLAB_LOG_DIR}
LOG_FILE="${SWANLAB_LOG_DIR}/log.txt"

# if you want to use swanlab or wandb tracking
# try to modify the `trainer.logger=['console','swanlab']` below

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
python3 -m recipe.On_Policy_SFT.main_on_policy_sft \
    data.train_files=$TRAIN_FILE \
    data.val_files="$VAL_FILE_LIST" \
    data.train_batch_size=32 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_GEN_LENGTH} \
    actor_rollout_ref.model.path=${BACKBONE_PATH} \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.enable_train_temperature=${ENABLE_TRAIN_TEMP} \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=False \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=2 \
    trainer.default_hdfs_dir=null \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    trainer.validation_data_dir=${OUTPUT_DIR}/rollout_eval_data \
    trainer.test_freq=20 \
    +trainer.log_freq=1 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.total_epochs=2 | tee ${LOG_FILE}
