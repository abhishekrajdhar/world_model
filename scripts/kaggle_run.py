"""Package the repo as a Kaggle dataset + GPU kernel, push it, and fetch results.

    python scripts/kaggle_run.py prepare     # build/refresh the source dataset
    python scripts/kaggle_run.py push        # push + run the GPU kernel
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py fetch       # download outputs into results/main_kaggle/

Kaggle kernels run headless, so the entry point is a plain script (kernel type "script").
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".kaggle_stage"
DATASET_SLUG = "rih-world-model-src"
KERNEL_SLUG = "rih-world-model-main"


def _user() -> str:
    cfg = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))
    return cfg["username"]


def _run(cmd: list[str], check=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip(), flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r


def prepare():
    user = _user()
    src = STAGE / "src_ds"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    for d in ["src", "configs", "scripts"]:
        shutil.copytree(ROOT / d, src / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "PREREGISTRATION.md", src / "PREREGISTRATION.md")
    meta = {"title": "RIH world model source", "id": f"{user}/{DATASET_SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    (src / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    existing = _run(["kaggle", "datasets", "list", "-m", "-s", DATASET_SLUG], check=False)
    if DATASET_SLUG in existing.stdout:
        _run(["kaggle", "datasets", "version", "-p", str(src), "-m", "update", "-d"])
    else:
        _run(["kaggle", "datasets", "create", "-p", str(src), "-d"])
    print("dataset ready; wait ~1 min for Kaggle to finish processing before push")


def push(config="main", seeds=(0, 1, 2)):
    user = _user()
    kdir = STAGE / "kernel"
    kdir.mkdir(parents=True, exist_ok=True)
    entry = f'''# Auto-generated Kaggle entry point (RIH experiment)
import os, subprocess, sys, shutil, json, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "minigrid==3.1.0", "gymnasium"], check=False)
SRC = "/kaggle/input/{DATASET_SLUG}"
WORK = "/kaggle/working/world_model"
os.makedirs(WORK, exist_ok=True)
for d in ["src", "configs", "scripts"]:
    shutil.copytree(os.path.join(SRC, d), os.path.join(WORK, d), dirs_exist_ok=True)
for d in ["results", "checkpoints", "experiments", "figures", "logs"]:
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, round(p.total_memory/1024**3, 2), "GB", "cc", f"{{p.major}}.{{p.minor}}")
    print("CUDA runtime", torch.version.cuda)

t0 = time.time()
for s in {list(seeds)}:
    subprocess.run([sys.executable, "scripts/run_experiment.py",
                    "--config", "{config}", "--seed", str(s)], check=True)
    print(f"seed {{s}} done at {{time.time()-t0:.0f}}s", flush=True)
subprocess.run([sys.executable, "scripts/analyze.py", "--config", "{config}",
                "--seeds", *[str(x) for x in {list(seeds)}]], check=True)

# keep outputs small: ship results, drop bulky checkpoints/trajectories
shutil.make_archive("/kaggle/working/results_{config}", "zip", os.path.join(WORK, "results"))
print("total runtime", time.time() - t0, "s")
'''
    (kdir / "kernel_main.py").write_text(entry)
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "RIH world model main",
        "code_file": "kernel_main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    _run(["kaggle", "kernels", "push", "-p", str(kdir)])


def status():
    _run(["kaggle", "kernels", "status", f"{_user()}/{KERNEL_SLUG}"], check=False)


def fetch():
    out = ROOT / "results" / "main_kaggle"
    out.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", f"{_user()}/{KERNEL_SLUG}", "-p", str(out)], check=False)
    print("downloaded to", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"prepare": prepare, "push": push, "status": status, "fetch": fetch}[cmd]()
