#!/usr/bin/env python3
"""Read kern bench reports: one report's scenarios and cost, or an A/B's verdict.

`show REPORT` prints every scenario's p50 and the cost line; `ops REPORT` each op's share of
the cost, decode and prefill apart. `ab DIR` reads the A/B that
loop/ab ran into DIR (a-<i>.json / b-<i>.json over the workload, ga-<i>.json / gb-<i>.json
over the guard), takes each side's median across pairs, and prints the change in cost, the
scenarios that moved, the guard's worst, and last a one-line verdict loop/land parses.
"""

import json
import statistics
import sys
from pathlib import Path

GAIN = -0.5  # percent of cost a candidate must save to count as a win
GUARD = 2.0  # percent any guard scenario may slow down


def load(path):
    return json.loads(Path(path).read_text())


def p50s(report):
    return {s["scenario"]["id"]: s["graph"]["stats"]["p50"] / 1e3 for s in report["scenarios"]}


def cost(report):
    return report["cost"]["total_us"] / 1e6


def ops(report, n=6):
    return " · ".join(f"{o['op']} {100 * o['share']:.1f}%" for o in report["cost"]["by_op"][:n])


def show(path):
    r = load(path)
    for k, v in p50s(r).items():
        print(f"{k:32s} {v:9.3f} ms")
    print(f"cost {cost(r):.3f} s · {ops(r)}")


def breakdown(path):
    """Where the cost goes: each op's weighted GPU seconds, split by decode and prefill."""
    r = load(path)
    seconds = {}
    for s in r["scenarios"]:
        weight, kind = s["scenario"]["weight"], s["program"]
        for c in s["calls"]:
            key = (c["op"], kind)
            seconds[key] = seconds.get(key, 0) + weight * c["in_program"]["stats"]["p50"] / 1e6
    total = sum(seconds.values())
    rows = sorted(seconds.items(), key=lambda kv: -kv[1])
    print(f"{'op':24s} {'program':8s} {'GPU s':>9s} {'share':>6s}")
    for (op, kind), s in rows:
        if s / total >= 0.002:
            print(f"{op:24s} {kind:8s} {s:9.1f} {100 * s / total:5.1f}%")
    print(f"total of calls {total:.1f} s (cost {cost(r):.1f} s; the rest is gaps between kernels)")


def median_side(files):
    reports = [load(f) for f in files]
    ids = p50s(reports[0])
    return (
        statistics.median(cost(r) for r in reports),
        {k: statistics.median(p50s(r)[k] for r in reports) for k in ids},
    )


def pct(a, b):
    return 100 * (b / a - 1)


def ab(d):
    d = Path(d)
    side = lambda prefix: median_side(sorted(d.glob(f"{prefix}-*.json")))
    (ca, sa), (cb, sb) = side("a"), side("b")
    _, ga = side("ga")
    _, gb = side("gb")
    moved = sorted(((pct(sa[k], sb[k]), k) for k in sa), key=lambda x: x[0])
    for change, k in moved:
        if abs(change) >= 0.5:
            print(f"{k:32s} {sa[k]:9.3f} -> {sb[k]:9.3f} ms  {change:+6.2f}%")
    worst = max(((pct(ga[k], gb[k]), k) for k in ga), key=lambda x: x[0])
    change = pct(ca, cb)
    win = change <= GAIN and worst[0] <= GUARD
    print(f"guard worst {worst[1]} {worst[0]:+.2f}% (limit +{GUARD}%)")
    print(
        json.dumps(
            {"a": round(ca, 3), "b": round(cb, 3), "change_pct": round(change, 3),
             "guard_worst": worst[1], "guard_pct": round(worst[0], 3), "win": win}
        )
    )


if __name__ == "__main__":
    {"show": show, "ops": breakdown, "ab": ab}[sys.argv[1]](*sys.argv[2:])
