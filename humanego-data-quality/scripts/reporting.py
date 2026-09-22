from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def create_plots(df: pd.DataFrame, episode_df: pd.DataFrame, out_dir: Path) -> List[str]:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    made: List[str] = []

    for col, title, xlabel in [
        ("brightness", "Image brightness distribution", "Mean grayscale value"),
        ("blur_laplacian", "Blur-score distribution", "Variance of Laplacian"),
        ("entropy", "Image entropy distribution", "Entropy (bits)"),
        ("mask_arm_coverage", "Arm-mask coverage", "Fraction of image"),
        ("mask_objects_mean_coverage", "Object-mask coverage", "Mean fraction of image"),
    ]:
        if col in df and pd.to_numeric(df[col], errors="coerce").notna().sum() >= 3:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            plt.figure(figsize=(7, 4))
            plt.hist(vals, bins=min(40, max(10, int(np.sqrt(len(vals))))))
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Frames")
            p = plots / f"{col}.png"
            _savefig(p)
            made.append(str(p.relative_to(out_dir)))

    if not episode_df.empty:
        top = episode_df.sort_values("frames", ascending=False).head(30)
        plt.figure(figsize=(10, max(4, 0.25 * len(top))))
        plt.barh(top["episode"].astype(str), top["frames"])
        plt.gca().invert_yaxis()
        plt.title("Frames per episode")
        plt.xlabel("Frames")
        p = plots / "frames_per_episode.png"
        _savefig(p)
        made.append(str(p.relative_to(out_dir)))

    # 2-D appearance PCA
    if "pca_x" in df and pd.to_numeric(df["pca_x"], errors="coerce").notna().sum() >= 3:
        plt.figure(figsize=(7, 6))
        episodes = list(dict.fromkeys(df["episode"].astype(str).tolist()))
        for ep in episodes[:20]:
            s = df[df["episode"].astype(str) == ep]
            x = pd.to_numeric(s["pca_x"], errors="coerce")
            y = pd.to_numeric(s["pca_y"], errors="coerce")
            plt.scatter(x, y, s=12, alpha=0.65, label=ep)
        plt.title("Lightweight appearance PCA (sampled frames)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if len(episodes) <= 12:
            plt.legend(fontsize=7)
        p = plots / "appearance_pca.png"
        _savefig(p)
        made.append(str(p.relative_to(out_dir)))

    # XY trajectory plots for a few entities.
    xyz_prefixes = sorted({c[:-2] for c in df.columns if c.endswith("_x") and (c.startswith("hand_") or c.startswith("camera") or c.startswith("object_"))})
    for prefix in xyz_prefixes[:8]:
        xcol, ycol = f"{prefix}_x", f"{prefix}_y"
        if xcol not in df or ycol not in df:
            continue
        if pd.to_numeric(df[xcol], errors="coerce").notna().sum() < 3:
            continue
        plt.figure(figsize=(7, 6))
        episodes = list(dict.fromkeys(df["episode"].astype(str).tolist()))
        for ep in episodes[:20]:
            s = df[df["episode"].astype(str) == ep]
            x = pd.to_numeric(s[xcol], errors="coerce")
            y = pd.to_numeric(s[ycol], errors="coerce")
            plt.plot(x, y, linewidth=1, alpha=0.7)
        plt.title(f"{prefix} XY trajectories")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        p = plots / f"{prefix}_xy.png"
        _savefig(p)
        made.append(str(p.relative_to(out_dir)))

    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    if speed_cols:
        plt.figure(figsize=(8, 4.5))
        labels, vals = [], []
        for c in speed_cols[:10]:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v):
                labels.append(c.replace("_speed", ""))
                vals.append(v.to_numpy())
        if vals:
            plt.boxplot(vals, tick_labels=labels, showfliers=False)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("m/s")
            plt.title("Translation-speed distributions")
            p = plots / "speed_boxplot.png"
            _savefig(p)
            made.append(str(p.relative_to(out_dir)))
        else:
            plt.close()

    return made


def _table_html(df: pd.DataFrame, cols: List[str], n: int = 20) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    available = [c for c in cols if c in df.columns]
    view = df[available].head(n).copy()
    return view.to_html(index=False, escape=True, classes="data-table", border=0, na_rep="")


def write_html_report(
    out_dir: Path,
    summary: Dict[str, Any],
    episode_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    plot_paths: List[str],
) -> Path:
    out = out_dir / "report.html"
    sev_counts = summary.get("issue_counts", {})
    score = float(summary.get("quality_score", 0))
    if score >= 90:
        score_label = "Clean / inspect warnings"
    elif score >= 80:
        score_label = "Usable with review"
    elif score >= 65:
        score_label = "Quality likely affects training"
    else:
        score_label = "Diagnosis recommended before training"

    top_episodes = episode_df.sort_values(["errors", "warnings"], ascending=False) if not episode_df.empty else episode_df
    warning_issues = issues_df[issues_df["severity"].isin(["error", "warning"])] if not issues_df.empty else issues_df
    warning_issues = warning_issues.head(80)

    cards = [
        ("Quality score", f"{score:.1f}/100", score_label),
        ("Episodes", str(summary.get("episodes", 0)), "recordings audited"),
        ("Frames", f"{summary.get('frames', 0):,}", f"sampled images: {summary.get('sampled_images', 0):,}"),
        ("Errors / warnings", f"{sev_counts.get('error',0)} / {sev_counts.get('warning',0)}", f"info: {sev_counts.get('info',0)}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html.escape(a)}</div><div class="value">{html.escape(b)}</div><div class="sub">{html.escape(c)}</div></div>'
        for a, b, c in cards
    )
    plots_html = "".join(
        f'<figure><img src="{html.escape(p)}" alt="{html.escape(Path(p).stem)}"><figcaption>{html.escape(Path(p).stem.replace("_", " "))}</figcaption></figure>'
        for p in plot_paths
    )

    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HumanEgo Data Quality Report</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--text:#1d2433;--muted:#667085;--border:#e4e7ec;--accent:#344054}}
body{{font-family:Inter,system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:28px}}
.wrap{{max-width:1180px;margin:auto}} h1{{margin:0 0 6px}} .muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}}
.card,section{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}}
.label,.sub{{color:var(--muted);font-size:13px}} .value{{font-size:28px;font-weight:700;margin:5px 0}}
section{{margin:14px 0}} .plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}}
figure{{margin:0}} img{{max-width:100%;border-radius:8px}} figcaption{{font-size:12px;color:var(--muted);margin-top:4px}}
.data-table{{border-collapse:collapse;width:100%;font-size:13px}} .data-table th,.data-table td{{border-bottom:1px solid var(--border);padding:8px;text-align:left;vertical-align:top}}
.data-table th{{background:#f9fafb;position:sticky;top:0}} .scroll{{overflow:auto;max-height:520px}}
code{{background:#f2f4f7;padding:2px 5px;border-radius:4px}}
</style></head><body><div class="wrap">
<h1>HumanEgo Data Quality Report</h1>
<div class="muted">Generated for <code>{html.escape(str(summary.get('dataset','HumanEgo')))}</code>. Score is heuristic and threshold-dependent.</div>
<div class="grid">{cards_html}</div>
<section><h2>Episode triage</h2><div class="scroll">{_table_html(top_episodes, ['episode','frames','duration_s','errors','warnings','near_duplicate_ratio','mean_brightness','mean_blur_laplacian','mean_effective_fps'], 60)}</div></section>
<section><h2>Issues requiring review</h2><div class="scroll">{_table_html(warning_issues, ['severity','category','episode','frame','metric','value','threshold','message'], 80)}</div></section>
<section><h2>Visual distributions</h2><div class="plots">{plots_html}</div></section>
<section><h2>Interpretation</h2><p>Start with episodes containing concentrated geometry/temporal errors. Treat image and duplicate heuristics as triage signals rather than automatic deletion rules. For robot demonstrations, preserve temporal continuity and prefer reviewing or recollecting a whole bad segment/episode.</p></section>
</div></body></html>"""
    out.write_text(content, encoding="utf-8")
    return out
