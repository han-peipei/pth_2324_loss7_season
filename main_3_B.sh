#!/bin/bash
set -euo pipefail
path=/kaggle/working/pth_2324_loss7_season/
mkdir -p "$path"
# 可选：线程数（Kaggle 不一定有 64 核）
# export OMP_NUM_THREADS=8

# 这个脚本所在目录（也就是仓库根目录/或你的代码目录）
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$PYTHONPATH"

# ✅ 入口脚本：用相对路径或用 ROOT 拼路径
ENTRY="$ROOT/data_main2.py"

# ✅ 日志：写到可写目录
LOG="$path/data_main2.log"

python -u "$ENTRY" 2>&1 | tee "$LOG"