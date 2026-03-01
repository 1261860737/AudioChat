#!/bin/bash
# AudioChat Pipeline Runner
# 用法: ./run_s2s.sh --audio <wav> [--output-dir <dir>]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ditx/llm/voice/3D-Speaker/.3ds/bin/python3"
PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/CosyVoice"

export PYTHONPATH

cd "$SCRIPT_DIR"

if [ "$1" = "s2s" ] || [ -z "$1" ]; then
    $PYTHON scripts/offline_pipeline_s2s.py "$@"
elif [ "$1" = "s2t" ]; then
    shift
    $PYTHON scripts/offline_pipeline.py "$@"
else
    echo "用法: ./run.sh [s2s|s2t] --audio <wav> [--output-dir <dir>]"
    echo "  s2s: 语音输出模式（默认）"
    echo "  s2t: 文本输出模式"
    exit 1
fi
