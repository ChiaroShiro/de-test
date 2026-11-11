#!/usr/bin/env bash
# 用 verl 对 Qwen3-30B-A3B 做 GRPO 强化学习（GSM8K 数学题），每一步保存一个 checkpoint，
# 目的是得到"相邻两步"的权重，用来复现差量（delta）权重同步的实验。
#
# 一般由 run_all.sh 调用；也可以单独运行（在 verl 容器里）：
#   冒烟测试：2 步、不存盘     SMOKE=1 bash run_grpo_qwen3_30b.sh
#   正式运行：每步存盘          bash run_grpo_qwen3_30b.sh
#
# 环境变量（run_all.sh 会自动设置）：
#   MODEL_30B  模型目录            WORK_DIR  工作目录（默认 /data/rl_delta）
#   STEPS      正式训练步数（默认 30）     KEEP  保留最后几个 checkpoint（默认 4）
#   PARAM_OFFLOAD / OPT_OFFLOAD  是否把参数 / 优化器状态放到主机内存（默认 True）
#   ROLLOUT    生成回答用的引擎：vllm（默认）或 sglang
#   LOG_FILE   日志文件路径
set -euo pipefail

MODEL_30B=${MODEL_30B:-/data/models/Qwen3-30B-A3B-Instruct-2507}
WORK_DIR=${WORK_DIR:-/data/rl_delta}
GSM8K_DIR=${GSM8K_DIR:-$WORK_DIR/gsm8k}
CKPT_ROOT=${CKPT_ROOT:-$WORK_DIR/ckpt}
LOG_DIR=${LOG_DIR:-$WORK_DIR/logs}
ROLLOUT=${ROLLOUT:-vllm}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPT_OFFLOAD=${OPT_OFFLOAD:-True}
KEEP=${KEEP:-4}

if [[ "${SMOKE:-0}" == "1" ]]; then
  EXP=smoke;  STEPS=2;               SAVE_FREQ=-1; BS=16; N=4     # 小批量跑 2 步，不存 checkpoint
else
  EXP=formal; STEPS=${STEPS:-30};    SAVE_FREQ=1;  BS=64; N=8     # 每步存一次，只保留最后 KEEP 个
fi
# 一步只做一次参数更新（ppo_mini_batch_size = train_batch_size），相邻 checkpoint 正好相差一次优化器更新

# 不同 verl 版本的配置项名字不完全一样：先在 verl 的配置文件里查一下，存在才传，避免 Hydra 报错
VERL_PKG=$(python3 -c "import verl, os; print(os.path.dirname(verl.__file__))")
CFG_DIR="$VERL_PKG/trainer/config"
has_key() { grep -rqsE "^[[:space:]]*$1:" "$CFG_DIR"; }
add_if() {        # add_if 配置项名 覆盖参数...
  local key=$1; shift
  if has_key "$key"; then ARGS+=("$@"); else echo "[提示] 当前 verl 版本没有配置项 $key，跳过：$*"; fi
}

if python3 -c "import flash_attn" 2>/dev/null; then
  RMPAD=True
else
  RMPAD=False; echo "[提示] 没有安装 flash_attn，关闭 use_remove_padding（训练会慢一些）"
fi

ARGS=(
  algorithm.adv_estimator=grpo
  data.train_files="$GSM8K_DIR/train.parquet"
  data.val_files="$GSM8K_DIR/test.parquet"
  data.train_batch_size=$BS
  data.max_prompt_length=512
  data.max_response_length=1024
  actor_rollout_ref.model.path="$MODEL_30B"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=$BS
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPT_OFFLOAD
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.name=$ROLLOUT
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.n=$N
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  'trainer.logger=["console"]'
  trainer.project_name=delta_sync
  trainer.experiment_name="qwen3_30b_a3b_grpo_$EXP"
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.test_freq=-1
  trainer.save_freq=$SAVE_FREQ
  trainer.total_training_steps=$STEPS
  trainer.default_local_dir="$CKPT_ROOT/$EXP"
)
add_if use_remove_padding      actor_rollout_ref.model.use_remove_padding=$RMPAD
add_if kl_loss_type            actor_rollout_ref.actor.kl_loss_type=low_var_kl
add_if entropy_coeff           actor_rollout_ref.actor.entropy_coeff=0
add_if use_kl_in_reward        algorithm.use_kl_in_reward=False
add_if filter_overlong_prompts data.filter_overlong_prompts=True
add_if truncation              data.truncation=error
add_if critic_warmup           trainer.critic_warmup=0
add_if val_before_train        trainer.val_before_train=False
add_if max_actor_ckpt_to_keep  trainer.max_actor_ckpt_to_keep=$KEEP
add_if resume_mode             trainer.resume_mode=disable      # 只存模型不存优化器，不能断点续训，每次从头跑

# checkpoint 只保存模型权重，不保存优化器状态（否则每个 checkpoint 会大好几倍）
if has_key save_contents; then
  ARGS+=('actor_rollout_ref.actor.checkpoint.save_contents=["model"]')
elif has_key contents; then
  ARGS+=('actor_rollout_ref.actor.checkpoint.contents=["model"]')
else
  echo "[警告] 找不到'只保存模型'的配置项，checkpoint 会连优化器一起保存，每个可能有 300 多 GB，注意磁盘"
fi

mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-$LOG_DIR/grpo_$EXP.log}
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
echo "[信息] 模型 $MODEL_30B | 实验 $EXP | 步数 $STEPS | rollout=$ROLLOUT | 参数offload=$PARAM_OFFLOAD 优化器offload=$OPT_OFFLOAD"
echo "[信息] 日志写入 $LOG_FILE"
python3 -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
