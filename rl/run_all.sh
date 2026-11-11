#!/usr/bin/env bash
# 一键完成：环境检查 -> 准备 GSM8K 数据 -> 冒烟测试 -> 正式 GRPO 训练 -> checkpoint 转 HuggingFace 格式
#
# 用法（在 verl 容器里）：
#   tmux new -s grpo                           # 正式训练要几个小时，强烈建议在 tmux 里运行
#   bash ~/de_xfer_bench/rl/run_all.sh         # 按顺序跑完全部阶段
#   bash ~/de_xfer_bench/rl/run_all.sh smoke   # 只跑某一个阶段：check / data / smoke / train / merge
#
# 可选环境变量：
#   VERL_DIR     verl 源码目录，默认 ~/verl
#   MODEL_30B    模型目录，默认在 /data/models 下自动查找名字含 30B-A3B 的目录
#   WORK_DIR     数据、checkpoint、日志的存放目录，默认 /data/rl_delta
#   TRAIN_STEPS  正式训练步数，默认 30
#   KEEP         保留最后几个 checkpoint，默认 4
#   FORCE=1      GPU 上有其他进程占用显存时也继续
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE=${1:-all}
export VERL_DIR=${VERL_DIR:-$HOME/verl}
export WORK_DIR=${WORK_DIR:-/data/rl_delta}
export GSM8K_DIR=$WORK_DIR/gsm8k
export CKPT_ROOT=$WORK_DIR/ckpt
export LOG_DIR=$WORK_DIR/logs
export KEEP=${KEEP:-4}
HF_OUT=$WORK_DIR/hf
TRAIN_STEPS=${TRAIN_STEPS:-30}
mkdir -p "$WORK_DIR" "$LOG_DIR"

say()  { echo -e "\n==================== $* ===================="; }
warn() { echo "[警告] $*" >&2; }
die()  { echo "[错误] $*" >&2; exit 1; }

find_model() {    # 在 /data/models 下找 Qwen3-30B-A3B，优先 Instruct-2507，排除量化版本
  if [[ -n "${MODEL_30B:-}" ]]; then echo "$MODEL_30B"; return 0; fi
  local list pick
  list=$(find /data/models -maxdepth 4 -name config.json -ipath "*30b-a3b*" -printf '%h\n' 2>/dev/null \
         | grep -viE "fp8|fp4|awq|gptq|gguf|int4|int8" | sort -u || true)
  [[ -n "$list" ]] || return 1
  pick=$(echo "$list" | grep -i "instruct-2507" | head -n1 || true)
  [[ -n "$pick" ]] || pick=$(echo "$list" | head -n1)
  echo "$pick"
}

prepare() {       # 每个阶段都需要：确定模型路径、决定 offload 方式
  MODEL_30B=$(find_model) || die "在 /data/models 下没找到 Qwen3-30B-A3B 模型，请用 MODEL_30B=模型目录 指定"
  export MODEL_30B
  ls "$MODEL_30B"/*.safetensors >/dev/null 2>&1 || die "$MODEL_30B 下没有 .safetensors 权重文件"
  local mem
  mem=$(free -g | awk '/^Mem:/{print $7}')
  if   (( mem >= 600 )); then export PARAM_OFFLOAD=True  OPT_OFFLOAD=True
  elif (( mem >= 300 )); then export PARAM_OFFLOAD=False OPT_OFFLOAD=True
  else die "主机可用内存只有 ${mem}G，放不下优化器状态（约 250G），请先释放内存"
  fi
  echo "模型: $MODEL_30B | 主机可用内存 ${mem}G -> 参数offload=$PARAM_OFFLOAD 优化器offload=$OPT_OFFLOAD"
}

stage_check() {
  say "1/5 环境检查"
  prepare
  if ! python3 -c "import verl" 2>/dev/null; then
    [[ -d "$VERL_DIR" ]] || die "找不到 verl 源码目录 $VERL_DIR，请用 VERL_DIR=目录 指定"
    echo "verl 还没安装，执行 pip install --no-deps -e $VERL_DIR"
    pip install --no-deps -e "$VERL_DIR"
  fi
  python3 - <<'PY'
import importlib, os, torch, verl
eng = os.environ.get("ROLLOUT", "vllm")
m = importlib.import_module(eng)
print("verl:", os.path.dirname(verl.__file__), f"| {eng}:", getattr(m, "__version__", "?"),
      "| torch:", torch.__version__, "| GPU 数:", torch.cuda.device_count())
assert torch.cuda.device_count() >= 4, "需要 4 张 GPU"
PY
  local busy
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', *' '$2 > 2000 {printf "GPU%s 已用 %sMiB  ", $1, $2}')
  if [[ -n "$busy" ]]; then
    [[ "${FORCE:-0}" == "1" ]] && warn "有 GPU 被占用：$busy（FORCE=1，继续）" \
      || die "有 GPU 被占用：$busy。先释放显存，或设置 FORCE=1 强制继续"
  fi
  local avail
  avail=$(df -BG --output=avail "$WORK_DIR" | tail -1 | tr -dc '0-9')
  echo "$WORK_DIR 所在磁盘剩余 ${avail}G"
  (( avail >= 700 )) || warn "磁盘剩余不足 700G（每个 checkpoint 约 120G，保留 $KEEP 个，再加转换后的文件）。可以设置 KEEP=2 减少占用"
}

stage_data() {
  say "2/5 准备 GSM8K 数据"
  if [[ -f "$GSM8K_DIR/train.parquet" && -f "$GSM8K_DIR/test.parquet" ]]; then
    echo "已存在，跳过：$GSM8K_DIR"
  else
    local script="$VERL_DIR/examples/data_preprocess/gsm8k.py" flag="--local_dir"
    [[ -f "$script" ]] || die "找不到 $script，请用 VERL_DIR=verl源码目录 指定"
    python3 "$script" --help 2>/dev/null | grep -q -- "--local_save_dir" && flag="--local_save_dir"
    if ! python3 "$script" "$flag" "$GSM8K_DIR"; then
      warn "直接访问 HuggingFace 失败，改用镜像 hf-mirror.com 重试"
      HF_ENDPOINT=https://hf-mirror.com python3 "$script" "$flag" "$GSM8K_DIR"
    fi
    [[ -f "$GSM8K_DIR/train.parquet" ]] || die "数据准备失败，$GSM8K_DIR 下没有 train.parquet"
  fi
  python3 -c "import pandas as pd; d=pd.read_parquet('$GSM8K_DIR/train.parquet'); print('训练集', len(d), '条，字段:', list(d.columns))"
}

stage_smoke() {
  say "3/5 冒烟测试（2 步，不保存 checkpoint）"
  SMOKE=1 LOG_FILE="$LOG_DIR/smoke.log" bash "$HERE/run_grpo_qwen3_30b.sh" \
    || die "冒烟测试失败。看最后 100 行：tail -n 100 $LOG_DIR/smoke.log"
  echo "冒烟测试通过。每一步耗时（秒）："
  grep -oE "timing_s/step:[0-9.]+" "$LOG_DIR/smoke.log" || echo "（日志里没找到 timing_s/step）"
}

stage_train() {
  say "4/5 正式训练（$TRAIN_STEPS 步，每步保存，保留最后 $KEEP 个）"
  if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
    warn "当前不在 tmux/screen 里，SSH 断开训练就会中断。10 秒后继续，按 Ctrl+C 取消"
    sleep 10
  fi
  STEPS=$TRAIN_STEPS LOG_FILE="$LOG_DIR/formal.log" bash "$HERE/run_grpo_qwen3_30b.sh" \
    || die "训练失败。看最后 100 行：tail -n 100 $LOG_DIR/formal.log"
  echo "已保存的 checkpoint："
  du -sh "$CKPT_ROOT"/formal/global_step_* 2>/dev/null || warn "没有找到 checkpoint"
}

stage_merge() {
  say "5/5 checkpoint 转成 HuggingFace 格式"
  local dirs d step out use_new=0
  mapfile -t dirs < <(ls -d "$CKPT_ROOT"/formal/global_step_* 2>/dev/null | sort -V)
  (( ${#dirs[@]} > 0 )) || die "没有找到 checkpoint：$CKPT_ROOT/formal/global_step_*"
  python3 -m verl.model_merger --help >/dev/null 2>&1 && use_new=1
  for d in "${dirs[@]}"; do
    step=${d##*_}; out="$HF_OUT/step_$step"
    if ls "$out"/*.safetensors >/dev/null 2>&1; then echo "已转换，跳过：$out"; continue; fi
    echo "转换 global_step_$step -> $out"
    if (( use_new )); then
      python3 -m verl.model_merger merge --backend fsdp --local_dir "$d/actor" --target_dir "$out"
    else      # 老版本 verl 的转换脚本
      python3 "$VERL_DIR/scripts/model_merger.py" merge --backend fsdp --local_dir "$d/actor" --target_dir "$out" \
        || python3 "$VERL_DIR/scripts/model_merger.py" --backend fsdp --hf_model_path "$MODEL_30B" \
                   --local_dir "$d/actor" --target_dir "$out"
    fi
  done
  echo "转换完成："
  du -sh "$HF_OUT"/step_*
}

case "$STAGE" in
  all)   stage_check; stage_data; stage_smoke; stage_train; stage_merge ;;
  check) stage_check ;;
  data)  stage_data ;;
  smoke) stage_check; stage_smoke ;;
  train) stage_check; stage_train ;;
  merge) prepare; stage_merge ;;
  *)     die "未知阶段 $STAGE，可选：all / check / data / smoke / train / merge" ;;
esac
say "完成（阶段：$STAGE）"
