"""Run the H5 OOD-repair experiment on Kaggle.

No world-model retraining is needed: the experiment only refits g_phi over features that were
already computed in the main run. Uploads those features as a dataset, then runs a kernel.

    python scripts/kaggle_ood.py prepare
    python scripts/kaggle_ood.py push
    python scripts/kaggle_ood.py status
    python scripts/kaggle_ood.py fetch
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
STAGE = ROOT / ".kaggle_stage_ood"
DATASET_SLUG = "rih-features-main"
KERNEL_SLUG = "rih-ood-repair"
TRANSFER_SLUG = "rih-transfer-h6"
ARCH_SLUG = "rih-cross-arch"
ARCH_CKPT_SLUG = "rih-cross-arch-ckpt"
SRC = ROOT / "results" / "main_kaggle" / "core" / "main"


def _user() -> str:
    return json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["username"]


def _run(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[-1500:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit("failed: " + " ".join(cmd))
    return r


def prepare(seeds=(0, 1, 2)):
    user = _user()
    d = STAGE / "ds"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for s in seeds:
        for f in [f"features_seed{s}.npz", f"feature_groups_seed{s}.json"]:
            src = SRC / f
            if not src.exists():
                raise SystemExit(f"missing {src}")
            shutil.copy(src, d / f)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "rih features main", "id": f"{user}/{DATASET_SLUG}",
         "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    total = sum(f.stat().st_size for f in d.iterdir()) / 1e6
    print(f"uploading {total:.0f} MB of precomputed features", flush=True)
    exists = _run(["kaggle", "datasets", "files", f"{user}/{DATASET_SLUG}"], check=False)
    if exists.returncode == 0 and "features_seed0" in exists.stdout:
        _run(["kaggle", "datasets", "version", "-p", str(d), "-m", "update"])
    else:
        _run(["kaggle", "datasets", "create", "-p", str(d)])


def _payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for x in ["src", "configs", "scripts"]:
            tf.add(ROOT / x, arcname=x,
                   filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def push(seeds=(0, 1, 2), slug=None, script="scripts/run_ood_repair.py",
         outdir="ood_repair", report="REPORT_H5.md", extra_args=None, need_features=True,
         ckpt_dataset=None, ckpt_dest=None):
    user = _user()
    k = STAGE / "kernel"
    k.mkdir(parents=True, exist_ok=True)
    slug = slug or KERNEL_SLUG
    entry = f'''# H5 OOD-repair experiment (no world-model retraining required)
import os, sys, subprocess, shutil, tarfile, base64, io, glob, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "minigrid==3.1.0", "gymnasium"],
               check=False)
WORK = "/kaggle/working/world_model"
os.makedirs(WORK, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("{_payload()}")), mode="r:gz") as tf:
    tf.extractall(WORK)
os.chdir(WORK)
os.makedirs("results/main", exist_ok=True)

NEED_FEATURES = {need_features}
if NEED_FEATURES:
    # Kaggle does not always mount at /kaggle/input/<slug>; search recursively instead.
    found = glob.glob("/kaggle/input/**/features_seed*.npz", recursive=True)
    if not found:
        for root, dirs, files in os.walk("/kaggle/input"):
            print("walk:", root, dirs[:5], files[:5], flush=True)
        raise SystemExit("features dataset not attached")
    src = os.path.dirname(found[0])
    print("features found in", src, flush=True)
    for f in glob.glob(os.path.join(src, "*")):
        if os.path.isfile(f) and (f.endswith(".npz") or f.endswith(".json")):
            shutil.copy(f, "results/main")
    print("inputs:", sorted(os.listdir("results/main")), flush=True)

CKPT_DEST = {ckpt_dest!r}
if CKPT_DEST:
    os.makedirs(CKPT_DEST, exist_ok=True)
    pts = glob.glob("/kaggle/input/**/*.pt", recursive=True)
    if not pts:
        raise SystemExit("checkpoint dataset not attached")
    for f in pts:
        shutil.copy(f, CKPT_DEST)
    print("restored", len(pts), "checkpoints into", CKPT_DEST, flush=True)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, "cc", f"{{p.major}}.{{p.minor}}", flush=True)
    if (p.major, p.minor) < (7, 0) and not os.environ.get("RIH_TORCH_PATCHED"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1",
                        "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
        os.environ["RIH_TORCH_PATCHED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

t0 = time.time()
r = subprocess.run([sys.executable, "{script}", "--config", "main",
                    "--seeds", *[str(x) for x in {list(seeds)}]] + {extra_args or []},
                   capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-4000:] if r.stderr else "")
os.makedirs("results/{outdir}", exist_ok=True)
open("results/{outdir}/{report}", "w").write(r.stdout)
shutil.make_archive("/kaggle/working/{outdir}", "zip", os.path.join(WORK, "results", "{outdir}"))
print("done in", time.time() - t0, "s")
if r.returncode != 0:
    raise SystemExit(r.returncode)
'''
    (k / "kernel_main.py").write_text(entry)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug.replace("-", " "),
        "code_file": "kernel_main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": (([f"{user}/{DATASET_SLUG}"] if need_features else [])
                            + ([f"{user}/{ckpt_dataset}"] if ckpt_dataset else [])),
        "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    _run(["kaggle", "kernels", "push", "-p", str(k)])


def pushtransfer():
    push(slug=TRANSFER_SLUG, script="scripts/run_transfer.py", outdir="transfer",
         report="REPORT_H6.md")


def pusharchsmoke():
    push(slug=ARCH_SLUG, script="scripts/run_cross_arch.py", outdir="cross_arch",
         report="REPORT_H7_smoke.md", seeds=(0,), need_features=False,
         extra_args=["--smoke", "--k", "2", "--steps", "300"])


def pusharch():
    """Analysis-only rerun: the 9 trained checkpoints are restored from a dataset, so
    train_arch() finds them cached and only feature extraction + analysis run."""
    push(slug=ARCH_SLUG, script="scripts/run_cross_arch.py", outdir="cross_arch",
         report="REPORT_H7.md", seeds=(0,), need_features=False,
         ckpt_dataset=ARCH_CKPT_SLUG, ckpt_dest="checkpoints/cross_arch",
         extra_args=["--k", "3", "--steps", "6000"])


def fetcharch():
    out = ROOT / "results" / "cross_arch_kaggle"
    out.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", f"{_user()}/{ARCH_SLUG}", "-p", str(out)], check=False)


def status():
    for sl in (KERNEL_SLUG, TRANSFER_SLUG, ARCH_SLUG):
        _run(["kaggle", "kernels", "status", f"{_user()}/{sl}"], check=False)


def fetchtransfer():
    out = ROOT / "results" / "transfer_kaggle"
    out.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", f"{_user()}/{TRANSFER_SLUG}", "-p", str(out)], check=False)


def fetch():
    out = ROOT / "results" / "ood_repair_kaggle"
    out.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", f"{_user()}/{KERNEL_SLUG}", "-p", str(out)], check=False)


if __name__ == "__main__":
    {"prepare": prepare, "push": push, "pushtransfer": pushtransfer, "status": status,
     "fetch": fetch, "fetchtransfer": fetchtransfer, "pusharchsmoke": pusharchsmoke,
     "pusharch": pusharch, "fetcharch": fetcharch}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
