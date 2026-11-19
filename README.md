# kern-rsi-demo

One GPU, a few coding agents, and a loop that makes Qwen3-4B cheaper to serve on that GPU,
keeping only what it measures and checks. Clone it, start it, come back to a history of
landed kernel changes, each with its cost before and after.

The model is a [kern](https://github.com/pegainfer-project/kern) manifest: the forward pass as
data (buffers, ops, the calls each program makes) over CUDA kernels compiled here. kern runs it
blindly, so every optimization is an edit to this repository, and every edit can be timed on
real traffic shapes and checked against the version before it.

## Start

An NVIDIA GPU from Ampere on (sm_80, 86, 89, 90, 100, 103, 120, 121) with 24 GB, a driver for
CUDA 13.3, Docker with the NVIDIA container toolkit, and an account for a coding agent CLI
(Claude Code by default; Codex and others through [humanize](https://github.com/humanfia/humanize)).

```sh
git clone https://github.com/pegainfer-project/kern-rsi-demo && cd kern-rsi-demo
git submodule update --init         # kern and FlashInfer, not FlashInfer's own submodules
docker build -t kern-rsi-demo .
docker run --gpus all -it --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" \
  -v "$HOME/.claude:/home/agent/.claude" -v "$HOME/.claude.json:/home/agent/.claude.json" \
  kern-rsi-demo
./start.sh 8        # hours; inside the container
```

`start.sh` fetches Qwen3-4B (8 GB), measures the starting manifest, then turns rounds until the
hours are spent. It runs in tmux: `tmux attach -t kern-rsi`, one window per agent. What landed
is in `state/HISTORY.md` and in `git log`. Knobs (agent, runs per round, round length, GPU)
are listed at the top of [`start.sh`](start.sh); with Codex, mount `~/.codex` and set
`KERN_RSI_AGENT=codex/<model>:high`.

## What turns

[`loop/README.md`](loop/README.md) is the loop: a coordinator session ranks the cost by op and
writes one brief per target, agents work on their targets in worktrees sharing the one GPU
through a lease, and each run lands on main only if it lowers the cost by 0.5%, slows no
guarded shape by 2%, and passes the numerics gate against main.

The cost is GPU seconds of [`workload.toml`](workload.toml): the decode and prefill shapes of
8.3 hours of a public agent service traced by kern-serve, scaled to Qwen3-4B's 32k context,
each weighted by how often it ran. Bring your own traffic by replacing that file.

## The starting point

On a GB300, the starting manifest prices the workload at 2481 GPU s: cuBLASLt GEMMs 32%,
FlashInfer prefill attention 31%, decode attention 20%, the rest small elementwise kernels at a
few percent of the card's bandwidth. Against transformers (teacher-forced, 16 steps over four
prompts and two chunk sizes), mean KL ≤ 2.3e-3, the only top-1 flips at exact ties
([`results/accuracy-sm103.json`](results/accuracy-sm103.json)).

## Layout

| path | what |
|---|---|
| `scripts/gen.py` | the manifest from the checkpoint's `config.json` and the built kernels |
| `scripts/attention.py`, `scripts/sampling.py` | FlashInfer's attention and sampling as kern ops |
| `scripts/build.py`, `kernels.toml` | the cubins, per SM, with their hashes |
| `scripts/accuracy.py` | kern against transformers |
| `source/` | the CUDA |
| `workload.toml` | the objective's traffic |
| `loop/` | the loop: briefs, gates, the orchestrator's scripts |
| `third_party/` | kern and FlashInfer, pinned |

Adapted from [kern-qwen3](https://github.com/pegainfer-project/kern-qwen3); provenance in
[`NOTICE`](NOTICE). Apache-2.0.
