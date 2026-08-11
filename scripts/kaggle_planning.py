"""Re-run ONLY the planning stage on Kaggle, reusing checkpoints from the main run.

Retraining 15 world models to fix a planning-task bug would waste hours of GPU. This uploads
the seed-0 checkpoints plus that seed's features as a dataset, then runs a short kernel.

    python scripts/kaggle_planning.py prepare
    python scripts/kaggle_planning.py push
    python scripts/kaggle_planning.py fetch
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".kaggle_stage_plan"
DATASET_SLUG = "rih-wm-ckpt-seed0"
KERNEL_SLUG = "rih-world-model-planning"
FAST_SLUG = "rih-world-model-planning-fast"
SRC = ROOT / "results" / "main_kaggle"


def _user() -> str:
    return json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["username"]


def _run(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[-1500:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit("failed: " + " ".join(cmd))
    return r


def prepare(seed=0):
    user = _user()
    d = STAGE / "ds"
    if d.exists():
        shutil.rmtree(d)
    (d / "checkpoints").mkdir(parents=True)
    (d / "results").mkdir(parents=True)
    for k in range(5):
        shutil.copy(SRC / "world_model" / "checkpoints" / "main" / f"wm_seed{seed}_member{k}.pt",
                    d / "checkpoints")
    for f in [f"features_seed{seed}.npz", f"feature_groups_seed{seed}.json",
              f"analysis_seed{seed}.json"]:
        shutil.copy(SRC / "core" / "main" / f, d / "results")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "RIH wm checkpoints seed0", "id": f"{user}/{DATASET_SLUG}",
         "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    exists = _run(["kaggle", "datasets", "files", f"{user}/{DATASET_SLUG}"], check=False)
    if exists.returncode == 0 and "checkpoints" in exists.stdout:
        _run(["kaggle", "datasets", "version", "-p", str(d), "-m", "update", "-r", "tar"])
    else:
        _run(["kaggle", "datasets", "create", "-p", str(d), "-r", "tar"])


def _payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for x in ["src", "configs", "scripts"]:
            tf.add(ROOT / x, arcname=x,
                   filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def push(seed=0, slug=None, episodes=60, n_samples=96, budget=80, fixed=None):
    user = _user()
    k = STAGE / "kernel"
    k.mkdir(parents=True, exist_ok=True)
    slug = slug or KERNEL_SLUG
    fixed_arg = (["--fixed"] + [str(x) for x in fixed]) if fixed else []
    entry = f'''# Planning-only re-run: reuses trained checkpoints, no retraining.
import os, sys, subprocess, shutil, tarfile, base64, io, glob, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "minigrid==3.1.0", "gymnasium"],
               check=False)
WORK = "/kaggle/working/world_model"
os.makedirs(WORK, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("{_payload()}")), mode="r:gz") as tf:
    tf.extractall(WORK)
os.chdir(WORK)
for d in ["results/main", "checkpoints/main", "figures", "logs", "experiments/main"]:
    os.makedirs(d, exist_ok=True)

DS = glob.glob("/kaggle/input/*/")
print("inputs:", DS, flush=True)
base = None
for cand in DS:
    if os.path.isdir(os.path.join(cand, "checkpoints")) or \\
       os.path.exists(os.path.join(cand, "checkpoints.tar")):
        base = cand
if base is None:
    raise SystemExit("checkpoint dataset not attached")
for name, dest in [("checkpoints", "checkpoints/main"), ("results", "results/main")]:
    p = os.path.join(base, name)
    if os.path.isdir(p):
        for f in os.listdir(p):
            shutil.copy(os.path.join(p, f), dest)
    else:
        with tarfile.open(os.path.join(base, name + ".tar")) as tf:
            tf.extractall("/tmp/x")
        for f in os.listdir(os.path.join("/tmp/x", name)):
            shutil.copy(os.path.join("/tmp/x", name, f), dest)
print("ckpts:", os.listdir("checkpoints/main"), flush=True)
print("results:", os.listdir("results/main"), flush=True)

import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, "cc", f"{{p.major}}.{{p.minor}}", flush=True)
    if (p.major, p.minor) < (7, 0) and not os.environ.get("RIH_TORCH_PATCHED"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1",
                        "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
        os.environ["RIH_TORCH_PATCHED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

t0 = time.time()
subprocess.run([sys.executable, "scripts/run_planning.py", "--config", "main",
                "--seed", "{seed}", "--episodes", "{episodes}", "--n-samples", "{n_samples}",
                "--budget", "{budget}"] + {fixed_arg}, check=True)
subprocess.run([sys.executable, "scripts/make_figures.py", "--config", "main",
                "--seed", "{seed}"], check=False)
shutil.make_archive("/kaggle/working/planning_results", "zip", os.path.join(WORK, "results"))
shutil.make_archive("/kaggle/working/planning_figures", "zip", os.path.join(WORK, "figures"))
print("done in", time.time() - t0, "s")
'''
    (k / "kernel_main.py").write_text(entry)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug.replace("-", " "),
        "code_file": "kernel_main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    _run(["kaggle", "kernels", "push", "-p", str(k)])


def pushfast():
    """Cheap parallel run: fewer episodes/samples/arms, guaranteed to finish quickly."""
    push(slug=FAST_SLUG, episodes=20, n_samples=48, budget=60, fixed=[3, 10, 30])


def status():
    for sl in (KERNEL_SLUG, FAST_SLUG):
        _run(["kaggle", "kernels", "status", f"{_user()}/{sl}"], check=False)


def fetchfast():
    out = ROOT / "results" / "planning_fast"
    out.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", f"{_user()}/{FAST_SLUG}", "-p", str(out)], check=False)


def fetch():
    out = ROOT / "results" / "planning_kaggle"
    out.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", f"{_user()}/{KERNEL_SLUG}", "-p", str(out)], check=False)


if __name__ == "__main__":
    {"prepare": prepare, "push": push, "pushfast": pushfast, "status": status,
     "fetch": fetch, "fetchfast": fetchfast}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
