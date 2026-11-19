# The loop

Coding agents make Qwen3-4B cheaper to serve on the GPU they run on, and the loop keeps only
what it can measure and check. Nothing in it is a prompt you write: the agents' brief
([`TASK.md`](TASK.md)), the coordinator's ([`DISPATCH.md`](DISPATCH.md)), the objective and
the gates are all in this directory, and `../start.sh` turns it.

```
          ┌──────────── state/main: bundle, bench.json, HISTORY.md ◄───────────┐
          ▼                                                                   │
   dispatch (1 session)  ──► briefs ──►  agents (N sessions, ralph_loop)  ──►  land
   reads cost by op,           one per     edit · make · check · ab,           squash, ab, check,
   history, notes/             target      commit each win                     fast-forward main
```

## One number

`cost` is what `kern bench` prices [`../workload.toml`](../workload.toml) at: every shape of a
real agent service's traffic (decode steps of 1-16 sequences over 1k-26k of context, prefill in
1024-row chunks), timed on this GPU from the manifest's own CUDA graphs, times how often it
ran. It is GPU seconds for the same traffic, so it moves only when kernels do. A change counts
when `loop/ab` (both manifests under one GPU lease, A B B A) shows it.

To run on your own traffic, replace `workload.toml`: `kern workload` turns a kern-serve trace
into one (`third_party/kern/docs/bench.md`), or write the shapes by hand.

## A round

| phase | what | who | time |
|---|---|---|---|
| dispatch | rank main's ops by GPU seconds, estimate headroom, split the work along kernel boundaries, write one brief per run | one fresh agent session (`hmz exec -f chat`) | ~5 min |
| optimize | each run in its own worktree and branch `agent/rN-<target>`: edit, `loop/make`, `loop/check`, `loop/ab`, commit each win | one agent per brief (`hmz exec -f ralph_loop`), until the deadline | `KERN_RSI_ROUND_MINUTES` (90) |
| land | squash each run onto main, rebuild, A/B against main, check, fast-forward on a win | `loop/land` | ~5 min a run |

Every run's `notes/<run>.md` lands with it, so the next round's dispatch and agents read what
was tried. The loop stops when its hours are spent or two rounds in a row land nothing.

## Gates

A run lands only when all three hold, measured by the orchestrator on main's own copy of the
scripts:

- **cost**: at least 0.5% lower than main's on `loop/ab`;
- **guard**: no scenario of [`guard.toml`](guard.toml) (every decode batch size to 16, every
  chunk size to 1024, short and long) more than 2% slower, so a kernel tuned to the traffic
  cannot regress what the traffic did not reach;
- **numerics**: `loop/check` PASS: `kern test` against main, every changed call replayed from
  main's inputs, the end-to-end logits within KL 1e-2, judged against the noise floor.

The starting manifest itself is held to transformers by `../scripts/accuracy.py`
(`../results/accuracy-sm103.json`: mean KL ≤ 2.3e-3, the only top-1 flips at exact ties).

## One GPU, several agents

Every command that uses the GPU runs under `loop/withgpu`, a lease on one lock file. Builds
(`nvcc`) and the agents' thinking run in parallel; benches and checks queue. Nobody's
measurement ever sees anybody else's kernels.

## Commands

| command | does | GPU |
|---|---|---|
| `loop/make [OUT]` | kernels for this GPU's SM, manifest, verify | no |
| `loop/bench BUNDLE OUT` | kern bench over the workload: scenarios and cost | lease |
| `loop/ab A B OUT [N]` | A B B A over workload and guard; the verdict | lease |
| `loop/check BUNDLE OUT` | kern test against main | lease |
| `loop/score.py ops REPORT` | cost by op, decode and prefill apart | no |
| `loop/agents status` | runs, their commits, which are live | no |
| `loop/agents tell NAME MSG` | a message at the top of a run's `ROUND.md` | no |
| `loop/round N` / `loop/land NAME` | one round / one landing, by hand | lease |

State lives in `state/`: `main/` (main's bundle and report), `HISTORY.md` (every landing),
`rounds/rN/` (briefs, dispatch log, summary), `runs/` (the agents' worktrees), `land/`.
