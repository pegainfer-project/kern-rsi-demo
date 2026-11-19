#!/usr/bin/env bash
# ./start.sh [HOURS]: run the loop on this machine's GPU for HOURS (default 8).
#
# Checks the machine, fetches Qwen3-4B, measures main, then turns rounds (loop/round) until the
# time is spent or two rounds in a row land nothing. Re-running picks up where it stopped. It
# runs in tmux session `kern-rsi`: `tmux attach -t kern-rsi` to watch, every agent a window.
#
#   KERN_RSI_AGENT          humanize agent spec       claude/claude-opus-5-5:high
#                           (codex/<model>:high, kimi/..., any CLI `hmz exec --help` lists)
#   KERN_RSI_LANES          agents per round          2
#   KERN_RSI_ROUND_MINUTES  agents' time per round    90
#   KERN_RSI_GPU            CUDA device               0
#   KERN_RSI_WEIGHTS        Qwen3-4B checkpoint       weights/Qwen3-4B (downloaded if absent)
#   KERN, NVCC              binaries                  from PATH
set -euo pipefail
cd "$(dirname "$0")"
hours=${1:-8}
if [[ -z ${TMUX:-} && -z ${KERN_RSI_FOREGROUND:-} ]]; then
  tmux has-session -t kern-rsi 2>/dev/null || tmux new-session -d -s kern-rsi -n main
  tmux new-window -t kern-rsi -n loop "KERN_RSI_FOREGROUND=1 ./start.sh $hours 2>&1 | tee -a state/start.log; exec bash"
  echo "start: running in tmux session kern-rsi (tmux attach -t kern-rsi)"
  exit 0
fi
mkdir -p state
source loop/env.sh

need() { command -v "$1" > /dev/null || { echo "start: $1 not found ($2)"; exit 1; }; }
need nvidia-smi "an NVIDIA driver"
need "$NVCC" "CUDA 13 nvcc"
need "$KERN" "kern: cargo build --release in third_party/kern"
need hmz "humanize: uv tool install git+https://github.com/humanfia/humanize.git@v0.2.2"
need tmux "tmux"
need "${AGENT%%/*}" "the agent CLI KERN_RSI_AGENT names, logged in"
grep -q "\"$SM\"" kernels.toml || { echo "start: $SM is not in kernels.toml's architectures"; exit 1; }
[[ $(git rev-parse --abbrev-ref HEAD) == main && -z $(git status --porcelain --untracked-files=no) ]] ||
  { echo "start: the loop lands on main: check out main with a clean tree"; exit 1; }
free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
((free >= 15000)) || { echo "start: GPU $GPU has $free MiB free; the loop needs 15000"; exit 1; }
git submodule update --init third_party/flashinfer third_party/kern
[[ -n $(git config user.email) ]] || git config user.email "agent@kern-rsi-demo.invalid"
[[ -n $(git config user.name) ]] || git config user.name "kern-rsi-demo agent"
[[ -f $WEIGHTS/config.json ]] || hf download Qwen/Qwen3-4B --local-dir "$WEIGHTS"

if [[ ! -f $STATE/main/bench.json ]]; then
  echo "start: measuring main on $(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)"
  loop/make "$STATE/main/bundle"
  loop/withgpu "$KERN" run --manifest "$STATE/main/bundle/model.json" --kernels "$STATE/main/bundle/kernels" \
    --weights "$WEIGHTS" --tokenizer "$WEIGHTS/tokenizer.json" --prompt "The capital of France is" --steps 16 2>/dev/null
  loop/bench "$STATE/main/bundle" "$STATE/main/bench" | tail -n 1
  cost=$(python3 -c "import json; print(round(json.load(open('$STATE/main/bench.json'))['cost']['total_us'] / 1e6, 3))")
  { printf '| # | commit | run | what | cost (GPU s) | change |\n|---|---|---|---|---|---|\n'
    printf '| 0 | %s | | main at the start | %s | |\n' "$(git log -1 --format=%h)" "$cost"; } > "$STATE/HISTORY.md"
fi

end=$(($(date +%s) + hours * 3600))
round_s=$((${KERN_RSI_ROUND_MINUTES:-90} * 60 + 900))
mkdir -p "$STATE/rounds"
n=$(($(find "$STATE/rounds" -mindepth 1 -maxdepth 1 -name "r*" | wc -l) + 1))
dry=0
while (($(date +%s) + round_s <= end && dry < 2)); do
  before=$(git rev-parse HEAD)
  loop/round "$n"
  [[ $(git rev-parse HEAD) == "$before" ]] && dry=$((dry + 1)) || dry=0
  n=$((n + 1))
done
echo "start: done after $((n - 1)) rounds"
cat "$STATE/HISTORY.md"
