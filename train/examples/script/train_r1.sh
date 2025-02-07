#!/bin/bash
set -euxo pipefail

HDFS_HOME=/gpfs/public/mmodal/users/vito
BASE_MODEL=Qwen2.5-Math-7B
RUN_NAME=Qwen2.5-Math-7B_ppo_from_base_math_lv35
WORKING_DIR=$HDFS_HOME/code/simpleRL-reason
GPU_PER_NODE=2

# Read distributed environment variables
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPU_PER_NODE}
NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

# NCCL config
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=12
# Instructed by Ray document to use this with openrlhf[vllm_latest].
export NCCL_P2P_DISABLE=1

export PYTHONPATH=$WORKING_DIR

# Ray specific settings
RAY_PORT=6379
RAY_HEAD_IP="$MASTER_ADDR:$RAY_PORT"
# Number of seconds to wait for the head node to be ready.
WAIT_INTERVAL_HEAD=30
# Number of seconds to wait for cluster to be ready before submitting the job.
WAIT_INTERVAL_SUBMIT=60

# Start Ray processes based on node rank
if [ "$NODE_RANK" -eq 0 ]; then
    echo "Starting HEAD node at $MASTER_ADDR"
    ray start --head \
        --node-ip-address=$MASTER_ADDR \
        --port=$RAY_PORT \
        --num-cpus=$NPROC_PER_NODE \
        --block &

    # Wait for Ray head node to be ready
    sleep $WAIT_INTERVAL_SUBMIT

    # Submit the Ray job from the head node
    # TODO: Change this.
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "${WORKING_DIR}", "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]}' \
      -- python3 $WORKING_DIR/train/openrlhf/cli/train_ppo_ray_box.py \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node $GPU_PER_NODE \
      --reward_num_nodes 0 \
      --reward_num_gpus_per_node 0 \
      --critic_num_nodes 1 \
      --critic_num_gpus_per_node $GPU_PER_NODE \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node $GPU_PER_NODE \
      --vllm_num_engines 16 \
      --vllm_tensor_parallel_size 1 \
      --colocate_actor_ref \
      --pretrain $HDFS_HOME/$BASE_MODEL \
      --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
      --micro_train_batch_size 2 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 2 \
      --rollout_batch_size 1024 \
      --temperature 0.6 \
      --n_samples_per_prompt 8 \
      --max_samples 100000 \
      --max_epochs 1 \
      --num_episodes 20 \
      --prompt_max_len 1024 \
      --generate_max_len 3000 \
      --zero_stage 3 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --critic_learning_rate 9e-6 \
      --init_kl_coef 0.01 \
      --prompt_data  $HDFS_HOME/data/math_level3to5_data_processed_with_qwen_prompt.json \
      --input_key input \
      --normalize_reward \
      --flash_attn \
      --gradient_checkpointing \
      --save_steps 4 \
      --load_checkpoint \
      --use_wandb 3e47658fc3138ec8580d195368499dd05ac5c735 \
      --wandb_project vito-r1 \
      --wandb_run_name $RUN_NAME \
      --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
      --max_ckpt_num 20000
else
    echo "Starting WORKER node $NODE_RANK"
    sleep $WAIT_INTERVAL_HEAD
    ray start --address "$RAY_HEAD_IP" \
        --num-cpus=$NPROC_PER_NODE \
        --block
fi