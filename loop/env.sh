# Sourced by every loop/ script. TREE is the checkout the command runs in (main or an agent's
# worktree); MAIN is the clone everything shares, STATE its loop state. Every default here is a
# variable start.sh's --help lists.
set -a
TREE=$(git rev-parse --show-toplevel)
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
LOOP=$MAIN/loop
STATE=${KERN_RSI_STATE:-$MAIN/state}
WEIGHTS=${KERN_RSI_WEIGHTS:-$MAIN/weights/Qwen3-4B}
GPU=${KERN_RSI_GPU:-0}
AGENT=${KERN_RSI_AGENT:-claude/claude-opus-5-5:high}
KERN=${KERN:-$(command -v kern || echo "$MAIN/third_party/kern/target/release/kern")}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
SM=${KERN_RSI_SM:-sm_$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader | tr -d .)}
# The served bounds: 32k of context, 1024-row prefill chunks, up to 16 sequences a decode step.
GEN_BOUNDS="--context 32768 --max-tokens 1024 --max-seqs 16"
set +a
mkdir -p "$STATE" 2>/dev/null || true
