# Task: Qwen3-4B on this GPU, fewer GPU seconds for the same traffic

Qwen3-4B (36 layers, 32 query / 8 KV heads of 128, BF16 weights, a 151936-token vocabulary) is
served by the kern runtime from a manifest: a typed list of buffers, ops and the calls each
program makes, over CUDA kernels this repository compiles. kern runs the manifest blindly; every
decision about the model's forward lives here, in `scripts/gen.py` and `source/`. `ROUND.md`
names your target for this run; this file covers what holds for every run.

## The objective

**`cost` from `loop/bench`, lower is better**: the GPU seconds the traffic in `workload.toml`
costs on this GPU. That traffic is a real agent service's steps (decode steps of 1-16
sequences over 1k-26k of context, prefill in 1024-row chunks), each shape weighted by how often
it ran. `loop/score.py ops REPORT` splits the cost by op, decode and prefill apart.

## The rules

- **Fixed**: the model's weights and their BF16 precision, the KV cache's layout (pages of 16
  slots, each `[K/V, kv_heads, 128]` in BF16), the served bounds in `loop/env.sh` (32k context,
  1024-row chunks, 16 sequences), the kern runtime (`third_party/kern`: read it, never change it).
- **Open**: every kernel and every launch: another tile, another kernel, a fused epilogue, fewer
  launches, a launch routed by batch size with `when: {var, min, max}` (`docs/manifest.md`).
  cuBLASLt (`extern:cublaslt_bf16_tn`) may be replaced where you can beat it, and kept where you
  cannot.
- **Numerics**: not bit-exactness, but every difference explained. A change must pass
  `loop/check`; its commit message says where its numerical difference from main comes from (a
  summation order, an f32 intermediate now bf16, ...).

## What you edit

- `source/`: kernel CUDA. A new module is a line in `kernels.toml`'s `[modules]`.
- `scripts/`: the ops and the calls (`gen.py`, `attention.py`, `sampling.py`) and the build
  (`build.py`: flags; kernels compile for the arch-specific target, `sm_103a`, `sm_120a`, where
  there is one).
- `notes/<run>.md` (your run's name is in `ROUND.md`): your memory (below). It lands on main
  with your commits, so later rounds read what you tried. Scratch under `work/`.

Never edit `loop/`, `workload.toml`, `kernels.toml`'s upstreams or anything under `third_party/`.
Never commit `build/`, `out/`, `work/`.

## Tools

All run from your worktree's root. The GPU is one card shared by every agent: commands that use
it take its lease through `loop/withgpu` themselves and wait when another agent holds it. Never
run GPU work outside a `loop/` command or `loop/withgpu`.

| Command | What it does | GPU | Time |
|---|---|---|---|
| `loop/make [OUT]` | build the kernels, generate the manifest, verify it (default `out/<sm>`) | none | ~20 s |
| `loop/bench BUNDLE OUT` | kern bench over `workload.toml`: every scenario's p50 and the cost | lease | ~15 s on GB300, minutes on smaller cards |
| `loop/ab A B OUT [N]` | A B B A on the lease over the workload and `loop/guard.toml`: the change in cost, the scenarios that moved, the guard's worst. **The number to quote** | lease | ~4x a bench a pair |
| `loop/check BUNDLE OUT` | **the numerics gate**: `kern test` against main's bundle, last line PASS / FAIL / INCONCLUSIVE | lease | ~1-3 min |
| `loop/score.py ops REPORT` | the cost by op | none | 0 s |

Main's bundle and report are `state/main/bundle` and `state/main/bench.json` in the main clone
(`$STATE/main/...`; `source loop/env.sh` sets `$STATE`): compare against them with `loop/ab`,
do not re-measure main. `state/` is the orchestrator's and read-only to you: write only in your
worktree, and take the GPU only through `loop/` commands, never with a lock of your own. The lease is shared, so a cycle's wall time includes waiting for it:
`edit -> loop/make -> loop/check -> loop/ab` is 5-15 minutes.

`kern bench` checks that a replay emits the tokens the one before it did: a kernel that is not
deterministic run to run fails the bench. Keep every kernel deterministic.

## Commits and wins

- One optimization a commit (`git commit -s`), subject `qwen3: <what is true after the commit,
  lowercase, no period>`. The body: `loop/ab`'s cost before -> after and the scenarios it moved,
  and `loop/check`'s verdict with where the numerical difference comes from.
- A win is `loop/check` PASS and `loop/ab` showing the cost at least 0.5% lower with no guard
  scenario more than 2% slower. Commit each win as soon as it passes; keep every commit building
  on its own. The orchestrator lands wins on main at the end of the round, by the same gates.
- This repository is public: no machine names, user names or absolute paths in what you commit.

## References (read-only)

- The kern runtime and manifest format: `third_party/kern/docs/` (`manifest.md`, `runtime.md`,
  `bench.md`, `test.md`) and `third_party/kern/crates/kern-manifest/src/types.rs`.
- FlashInfer at the pinned revision: `third_party/flashinfer/include/`. Attention and sampling
  are instantiated from its headers (`source/attention.cu`, `source/sampling.cu`).
- `nvidia-smi` and `nvcc --list-gpu-arch` say what card this is; tune for it, not for another.

## Boundaries

No network (no curl, pip, git fetch). Read only your worktree and the paths above. Never touch
processes you did not start.

## Time and sessions

`ROUND.md` has your deadline; the session is killed then, so commit each win when it passes.
Work until the deadline. If your target is closed (at its roofline, or fused as far as it
sensibly goes), write why in `notes/<run>.md`, then take the next most expensive op that `ROUND.md`
does not give to another run.

This runs as a loop of fresh sessions. Your only memory is the worktree:

- **`notes/<run>.md`**: read it first, and the other files in `notes/` from earlier rounds; update it before you stop: the current cost, what dominates
  your target, every idea tried with its measured result and why it was kept or reverted.
- **git**: commit each passing improvement on its own. Leave the tree clean.
