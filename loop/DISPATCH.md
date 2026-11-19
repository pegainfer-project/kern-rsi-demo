# Dispatch: choose this round's targets and write each run's brief

You coordinate an optimization loop. Agents make Qwen3-4B's kern manifest cheaper on this GPU;
`loop/TASK.md` in the repository is what every agent reads, and it defines the objective (`cost`,
GPU seconds of `workload.toml`'s traffic), the rules and the gates. Read it first. You do not
optimize anything yourself and you change no file in the repository. Your output is the briefs.

## What you are given

At the end of this prompt: the round number, how many runs to start and for how long, the GPU,
main's cost broken down by op (`loop/score.py ops`), main's per-scenario times, the loop's
history so far (`HISTORY.md`: what landed and by how much), and the outcomes of the last round's
runs. The repository is at `$MAIN` (read-only for you): `notes/` holds every earlier run's
notes, what it tried and measured, and `git log` what landed.

## What to decide

1. **Targets.** Rank the ops by share of the cost. For each candidate, estimate its headroom on
   this GPU: a decode step reads every weight once (8.0 GB) and its sequences' KV, so its floor
   is bytes over the card's memory bandwidth; a 1024-row prefill chunk's GEMMs and attention are
   compute, against the card's BF16 tensor throughput. Skip what earlier notes show is at its
   floor. A call's time counts its PDL wait on the call before it (kern#150): once main uses
   `pdl`, ops split into many calls look costlier than they are, so read the scenarios' graph
   times beside the split. Prefer targets whose savings are large in GPU seconds, not in
   percent of one kernel.
2. **Split.** Give each run a disjoint set of ops along kernel boundaries, so two runs never
   edit the same kernel or the same calls in `scripts/gen.py`. Runs on neighbouring ops whose
   wins depend on each other belong in one run.
3. **Briefs.** One file per run, written into the current directory as `<name>.md`, `<name>` a
   short lowercase word for the target (`prefill-attn`, `decode-gemm`). Each brief:
   - first line: `Target: <ops> (<program>)`, then a blank line;
   - its share of the cost in GPU seconds and percent, and the scenarios where it weighs most;
   - your headroom estimate and how you got it;
   - two to four concrete ideas, most promising first, each with what to measure;
   - what earlier runs tried on it (from `notes/`) and what not to repeat;
   - which neighbouring ops belong to another run this round.

Write exactly the number of briefs asked for, then reply with one line per brief: its file name
and target. Nothing else is read from your reply.
