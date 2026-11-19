#!/usr/bin/env python3
"""Compile every module in kernels.toml for one or more SMs and record what was built.

`build/<sm>/build.json` carries the compiler, the upstream revisions, each cubin's
SHA-256, and the facts the generator reads off the build instead of assuming them:
FlashInfer's attention parameter layouts, its merge kernel's symbol, and the
sampling chain's storage sizes and entry points.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = "-Ithird_party/flashinfer/include"
FLAGS = ("-std=c++17", "-O3", "--fmad=false", "--expt-relaxed-constexpr", "-Isource", INCLUDE)
# nvcc names internal-linkage symbols after a hash of the source file's path, so the same
# source compiled in two worktrees gives two cubins. Compiled from a path named by its
# content, a source gives the same bytes wherever the checkout is.
SOURCES = Path(tempfile.gettempdir()) / "kern-rsi-src"
SAMPLING_ENTRIES = (
    "OnlineSoftmaxFusedKernel",
    "RadixTopKRenormProbKernel_MultiCTA",
    "TopPRenormProbKernel",
    "MinPSamplingFromProbKernel",
)


def run(*command):
    return subprocess.check_output(command, cwd=ROOT, text=True)


def pinned(config):
    for name, upstream in config["upstreams"].items():
        actual = run("git", "-C", upstream["path"], "rev-parse", "HEAD").strip()
        if actual != upstream["revision"]:
            raise ValueError(f"{name}: expected {upstream['revision']}, got {actual}")


def probe(nvcc, source):
    """Compile and run a host program that prints JSON facts about the headers."""
    exe = ROOT / "build" / Path(source).stem
    exe.parent.mkdir(parents=True, exist_ok=True)
    run(nvcc, "-std=c++17", INCLUDE, source, "-o", str(exe))
    return json.loads(run(str(exe)))


def functions(cuobjdump, cubin):
    return dict(re.findall(r"Function ([^:]+):\s*([^\n]+)", run(cuobjdump, "-res-usage", str(cubin))))


def sampling_entries(found):
    entries = {
        tag: {"entry": entry, "static_shared": int(re.search(r"SHARED:(\d+)", usage)[1])}
        for tag in SAMPLING_ENTRIES
        for entry, usage in found.items()
        if tag in entry
    }
    if len(entries) != len(SAMPLING_ENTRIES):
        raise ValueError(f"expected one sampling entry per {SAMPLING_ENTRIES}, got {sorted(entries)}")
    return entries


def placed(source):
    """The source at its content-addressed path, written once and atomically."""
    data = (ROOT / source).read_bytes()
    path = SOURCES / hashlib.sha256(data).hexdigest()[:16] / Path(source).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return str(path)


def compile_sm(nvcc, cuobjdump, modules, sm):
    out = ROOT / "build" / sm
    out.mkdir(parents=True, exist_ok=True)
    # A bundle serves one SM, so it takes the arch-specific target where there is one
    # (sm_90a, sm_100a, sm_103a, sm_120a): wgmma, tcgen05 and TMA are open to its kernels.
    arch = f"{sm}a" if f"{sm}a" in run(nvcc, "--list-gpu-code").split() else sm

    def cubin(name, source):
        target = out / f"{name}.cubin"
        run(nvcc, "--cubin", *FLAGS, f"-arch={arch}", placed(source), "-o", str(target))
        return name, {"source": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    built = dict(cubin(name, source) for name, source in modules.items())
    merge = [e for e in functions(cuobjdump, out / "attention.cubin") if "PersistentVariableLengthMergeStatesKernel" in e]
    if len(merge) != 1:
        raise ValueError(f"expected one FlashInfer merge entry, got {merge}")
    return out, built, merge[0], sampling_entries(functions(cuobjdump, out / "sampling.cubin"))


def build(sms, nvcc):
    config = tomllib.loads((ROOT / "kernels.toml").read_text())
    pinned(config)
    supported = run(nvcc, "--list-gpu-code").split()
    unknown = [sm for sm in sms if sm not in config["architectures"] or sm not in supported]
    if unknown:
        raise ValueError(f"unsupported architectures {unknown}; kernels.toml lists {config['architectures']}")
    cuobjdump = str(Path(nvcc).resolve().with_name("cuobjdump"))
    compiler = run(nvcc, "--version").strip()
    sizes = probe(nvcc, "source/sampling_metadata.cu")
    attention = probe(nvcc, "source/attention_metadata.cu")
    for sm in sms:
        out, modules, merge, entries = compile_sm(nvcc, cuobjdump, config["modules"], sm)
        record = {
            "architecture": sm,
            "compiler": compiler,
            "upstreams": {k: v["revision"] for k, v in config["upstreams"].items()},
            "modules": modules,
            "attention": attention | {"merge_entry": merge},
            "sampling": {"sizes": sizes, "entries": entries},
        }
        (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{sm}: {len(modules)} modules", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sm", nargs="+", required=True, help="SM names (sm_89 ...), or all")
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    args = parser.parse_args()
    if not args.nvcc:
        parser.error("nvcc is required; pass --nvcc or put it on PATH")
    archs = tomllib.loads((ROOT / "kernels.toml").read_text())["architectures"]
    build(archs if args.sm == ["all"] else args.sm, args.nvcc)
