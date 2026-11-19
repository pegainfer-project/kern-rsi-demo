# Sourced by every loop/ script. TREE is the checkout the command runs in (main, a landing's
# worktree, or an agent's clone, which names main in `kern-rsi.main`); MAIN is the clone
# everything shares, STATE its loop state. Every default here is a
# variable start.sh's --help lists.
set -a
TREE=$(git rev-parse --show-toplevel)
MAIN=$(git config kern-rsi.main || (cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd))
LOOP=$MAIN/loop
STATE=${KERN_RSI_STATE:-$MAIN/state}
WEIGHTS=${KERN_RSI_WEIGHTS:-$MAIN/weights/Qwen3-4B}
GPU=${KERN_RSI_GPU:-0}
AGENT=${KERN_RSI_AGENT:-claude/claude-opus-5-5:high}
KERN=${KERN:-$(command -v kern || echo "$MAIN/third_party/kern/target/release/kern")}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
SM=${KERN_RSI_SM:-sm_$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader | tr -d .)}
# The served bounds: 8k of context, 1024-row prefill chunks, up to 16 sequences a decode step.
GEN_BOUNDS="--context 8192 --max-tokens 1024 --max-seqs 16"
set +a
mkdir -p "$STATE" 2>/dev/null || true
