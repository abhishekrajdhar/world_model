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

    existing = _run(["kaggle", "datasets", "files", f"{user}/{DATASET_SLUG}"], check=False)
    if existing.returncode == 0 and "PREREGISTRATION" in existing.stdout:
        _run(["kaggle", "datasets", "version", "-p", str(src), "-m", "update", "-r", "tar"])
    else:
        _run(["kaggle", "datasets", "create", "-p", str(src), "-r", "tar"])
    print("dataset ready; wait ~1 min for Kaggle to finish processing before push")


def _embedded_payload() -> str:
    """tar.gz the source tree and base64 it, so the kernel is fully self-contained.

    Embedding avoids depending on Kaggle's dataset mount, which silently failed to appear
    at /kaggle/input/<slug> even with a 'ready' dataset and correct dataset_sources.
    """
    import base64
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for d in ["src", "configs", "scripts"]:
            tf.add(ROOT / d, arcname=d,
                   filter=lambda ti: None if ("__pycache__" in ti.name or
                                              ti.name.endswith(".pyc")) else ti)
    return base64.b64encode(buf.getvalue()).decode()


def push(config="main", seeds=(0, 1, 2)):
    user = _user()
    kdir = STAGE / "kernel"
    kdir.mkdir(parents=True, exist_ok=True)
    payload = _embedded_payload()
    print(f"embedded source payload: {len(payload)/1024:.0f} KB")
    entry = f'''# Auto-generated Kaggle entry point (RIH experiment) -- source embedded inline
import os, subprocess, sys, shutil, json, time, base64, io, tarfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "minigrid==3.1.0", "gymnasium"], check=False)
WORK = "/kaggle/working/world_model"
os.makedirs(WORK, exist_ok=True)
PAYLOAD = "{payload}"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tf:
    tf.extractall(WORK)
assert os.path.isdir(os.path.join(WORK, "src")), os.listdir(WORK)
print("unpacked:", sorted(os.listdir(WORK)))
for d in ["results", "checkpoints", "experiments", "figures", "logs"]:
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, round(p.total_memory/1024**3, 2), "GB", "cc", f"{{p.major}}.{{p.minor}}")
    print("CUDA runtime", torch.version.cuda)
    # Kaggle may allocate a Pascal P100 (sm_60); modern torch wheels ship sm_70+ only.
    # Install a Pascal-capable build once, then re-exec.
    if (p.major, p.minor) < (7, 0) and not os.environ.get("RIH_TORCH_PATCHED"):
        print("GPU capability below torch support: installing Pascal-compatible torch",
              flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.5.1", "--index-url",
                        "https://download.pytorch.org/whl/cu121"], check=False)
        os.environ["RIH_TORCH_PATCHED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

t0 = time.time()
SEEDS = {list(seeds)}

def stage(name, cmd, required):
    """Run a stage. Optional stages must never take down the main result."""
    print(f"\\n===== {{name}} (t+{{time.time()-t0:.0f}}s) =====", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[{{'FATAL' if required else 'WARN'}}] stage {{name}} failed", flush=True)
        if required:
            raise SystemExit(1)
        return False
    return True

# --- required: train, extract features, analyse ---------------------------
for s in SEEDS:
    stage(f"experiment seed {{s}}",
          [sys.executable, "scripts/run_experiment.py", "--config", "{config}",
           "--seed", str(s)], required=True)
stage("analyze", [sys.executable, "scripts/analyze.py", "--config", "{config}",
                  "--seeds", *[str(x) for x in SEEDS]], required=True)
# archive the core result immediately, before the optional stages run
shutil.make_archive("/kaggle/working/results_core", "zip", os.path.join(WORK, "results"))
print("core results archived at t+%.0fs" % (time.time() - t0), flush=True)

# --- optional: ablation, planning, figures, report ------------------------
stage("ablation K", [sys.executable, "scripts/run_ablation.py", "--config", "{config}",
                     "--seed", str(SEEDS[0]), "--ks", "1", "3", "5"], required=False)
stage("planning", [sys.executable, "scripts/run_planning.py", "--config", "{config}",
                   "--seed", str(SEEDS[0]), "--episodes", "60", "--n-samples", "96",
                   "--budget", "80"], required=False)
for s in SEEDS:
    stage(f"figures seed {{s}}", [sys.executable, "scripts/make_figures.py",
                                 "--config", "{config}", "--seed", str(s)], required=False)

print("\\n===== REPORT =====", flush=True)
rep = subprocess.run([sys.executable, "scripts/make_report.py", "--config", "{config}"],
                     capture_output=True, text=True)
print(rep.stdout)
print(rep.stderr[-2000:] if rep.stderr else "")
open(os.path.join(WORK, "results", "REPORT.md"), "w").write(rep.stdout)

# drop bulky intermediates so the downloadable output stays manageable
import glob
for pat in ["results/*/features_seed*.npz", "results/*/ablation_K/features_*.npz",
            "experiments/*/*.npz"]:
    for f in glob.glob(os.path.join(WORK, pat)):
        os.remove(f)

shutil.make_archive("/kaggle/working/results_{config}", "zip", os.path.join(WORK, "results"))
shutil.make_archive("/kaggle/working/figures_{config}", "zip", os.path.join(WORK, "figures"))
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
        "dataset_sources": [],
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
