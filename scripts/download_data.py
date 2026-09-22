#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download HumanEgo Aria data from the public HuggingFace dataset Leo-TX/HumanEgo.

No token / login required — the dataset is public.

  --task   serve_bread | water_flowers | all   (default: serve_bread)
  --num    N  or  all                          (default: 1 — the first N recordings)
  --input-only   download inputs only (run preprocessing yourself); skip the
                 precomputed `preprocess/` output. Otherwise it is included and each
                 recording's `all_data.tar` is auto-extracted.
  --out    download destination                (default: ./data)
  --keep-tar     keep all_data.tar after extracting (default: delete to save space)

Examples
--------
    python scripts/download_data.py                                # 1 serve_bread recording, full output
    python scripts/download_data.py --task serve_bread --num 2     # first 2 serve_bread, with output
    python scripts/download_data.py --task all       --num all     # the entire dataset (large)
    python scripts/download_data.py --task water_flowers --num 5 --input-only
"""
import argparse
import os
import subprocess
import tarfile
import threading
import time

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Leo-TX/HumanEgo"
TASKS = ["serve_bread", "water_flowers"]

# Optional pretty UI (rich is a project dependency; fall back to plain prints if absent).
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                               DownloadColumn, TransferSpeedColumn, TimeRemainingColumn)
    _C = Console()
    HAS_RICH = True
except Exception:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------
def list_recordings(task, input_only):
    """{recording_path: total_bytes} for a task, sorted; honours --input-only."""
    try:
        tree = HfApi().list_repo_tree(REPO_ID, path_in_repo=f"{task}/aria",
                                      repo_type="dataset", recursive=True)
    except Exception:
        return {}
    sizes = {}
    for e in tree:
        size = getattr(e, "size", None)
        if size is None:                       # folder entry, not a file
            continue
        segs = e.path.split("/")
        vi = next((i for i, s in enumerate(segs) if s.endswith("_vrs")), None)
        if vi is None:
            continue
        if input_only and len(segs) > vi + 1 and segs[vi + 1] == "preprocess":
            continue
        rec = "/".join(segs[: vi + 1])
        sizes[rec] = sizes.get(rec, 0) + size
    return dict(sorted(sizes.items()))


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _run_threaded(fn):
    """Run fn() in a daemon thread; return (thread, err_holder)."""
    err = {}

    def _wrap():
        try:
            fn()
        except Exception as e:  # surfaced to the caller after join()
            err["e"] = e

    th = threading.Thread(target=_wrap, daemon=True)
    th.start()
    return th, err


# ---------------------------------------------------------------------------
# Pretty (rich) path — one byte-progress bar per phase per recording
# ---------------------------------------------------------------------------
def _bar(*extra):
    return Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                    *extra, TransferSpeedColumn(), TimeRemainingColumn(), console=_C)


def _download_recording(rec, out, input_only, total):
    ignore = [f"{rec}/preprocess/*"] if input_only else None
    th, err = _run_threaded(lambda: snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=out,
        allow_patterns=[f"{rec}/*"], ignore_patterns=ignore))
    recdir = os.path.join(out, rec)
    with _bar(DownloadColumn()) as p:
        t = p.add_task(f"[cyan]⬇ download[/]  {os.path.basename(rec)}", total=max(total, 1))
        while th.is_alive():
            p.update(t, completed=min(_dir_size(recdir), total))
            time.sleep(0.3)
        th.join()
        p.update(t, completed=max(total, 1))
    if "e" in err:
        raise err["e"]


def _extract_recording(tar_path, keep):
    out = os.path.dirname(tar_path)                       # the preprocess/ dir
    rec = os.path.basename(os.path.dirname(out))
    total = os.path.getsize(tar_path)                     # all_data.tar is uncompressed
    target = os.path.join(out, "all_data")               # tar unpacks into preprocess/all_data/
    th, err = _run_threaded(lambda: subprocess.run(
        ["tar", "-xf", tar_path, "-C", out], check=True))
    with _bar(DownloadColumn()) as p:
        t = p.add_task(f"[magenta]📦 extract [/]  {rec}", total=max(total, 1))
        while th.is_alive():
            p.update(t, completed=min(_dir_size(target), total))
            time.sleep(0.3)
        th.join()
        p.update(t, completed=max(total, 1))
    if "e" in err:
        raise err["e"]
    if not keep:
        os.remove(tar_path)


def run_rich(recs, sizes, out, input_only, keep, task_label):
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()                          # we render our own bars
    except Exception:
        pass
    grand = sum(sizes.values())
    mode = "inputs only" if input_only else "inputs + precomputed output"
    _C.print(Panel.fit(
        f"[bold]HumanEgo dataset[/]   task=[cyan]{task_label}[/]   "
        f"[cyan]{len(recs)}[/] recording(s)   {mode}\n"
        f"≈ [yellow]{grand / 1e9:.1f} GB[/]   →   [green]{os.path.abspath(out)}[/]",
        title="📥  HumanEgo Download", border_style="cyan"))
    os.makedirs(out, exist_ok=True)
    for i, rec in enumerate(recs, 1):
        _C.rule(f"[bold cyan][{i}/{len(recs)}][/] {rec}")
        _download_recording(rec, out, input_only, sizes.get(rec, 0))
        if not input_only:
            tar = os.path.join(out, rec, "preprocess", "all_data.tar")
            if os.path.exists(tar):
                _extract_recording(tar, keep)
    nxt = ""
    if input_only:
        t0 = recs[0].split("/")[0]
        nxt = ("\n\n[bold]Next[/] — preprocess it:\n"
               f"  [green]python -m preprocess.Preprocess "
               f"--mps_path {os.path.join(out, recs[0])} --task {t0}[/]")
    _C.print(Panel.fit(
        f"[bold green]✓ Done[/]  •  {len(recs)} recording(s)  •  "
        f"[green]{os.path.abspath(out)}[/]{nxt}", border_style="green"))


# ---------------------------------------------------------------------------
# Plain fallback (no rich)
# ---------------------------------------------------------------------------
def run_plain(recs, out, input_only, keep):
    import glob
    mode = "input only" if input_only else "input + precomputed output"
    print(f"Downloading {len(recs)} recording(s) ({mode}) -> {out}")
    os.makedirs(out, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=out,
                      allow_patterns=[f"{r}/*" for r in recs],
                      ignore_patterns=["*/preprocess/*"] if input_only else None)
    if not input_only:
        for tar in glob.glob(os.path.join(out, "**", "preprocess", "all_data.tar"), recursive=True):
            print(f"  extracting {os.path.relpath(tar, out)}")
            subprocess.run(["tar", "-xf", tar, "-C", os.path.dirname(tar)], check=True)
            if not keep:
                os.remove(tar)
    print("✅ done")
    if input_only:
        t0 = recs[0].split("/")[0]
        print(f"\nNext — preprocess: python -m preprocess.Preprocess "
              f"--mps_path {os.path.join(out, recs[0])} --task {t0}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS + ["all"], default="serve_bread")
    ap.add_argument("--num", default="1", help="how many recordings (first N), or 'all' (default: 1)")
    ap.add_argument("--input-only", action="store_true",
                    help="download inputs only (run preprocessing yourself); skip precomputed output")
    ap.add_argument("--out", default="./data", help="download destination (default: ./data)")
    ap.add_argument("--keep-tar", action="store_true",
                    help="keep all_data.tar after extracting (default: delete to save space)")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    num = None if str(args.num).lower() == "all" else int(args.num)

    sizes = {}
    for t in tasks:
        found = list(list_recordings(t, args.input_only).items())
        if not found:
            print(f"  [warn] no recordings found for task '{t}' (not uploaded yet?)")
        sizes.update(dict(found if num is None else found[:num]))
    recs = list(sizes.keys())
    if not recs:
        raise SystemExit("Nothing to download.")

    if HAS_RICH:
        run_rich(recs, sizes, args.out, args.input_only, args.keep_tar, args.task)
    else:
        run_plain(recs, args.out, args.input_only, args.keep_tar)


if __name__ == "__main__":
    main()
