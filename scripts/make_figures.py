"""Build every README figure.

    uv run --extra models --extra bench --extra viz python scripts/make_figures.py

Inputs come from ``sufficiency bench``: ``runs/estimate`` (results, cached features and
fitted estimators), ``runs/selective`` (per-question scores) and ``runs/recover``.
Outputs go to ``docs/assets/`` as ``<name>-light.*`` and ``<name>-dark.*``.

Colours come from a fixed palette checked for colour-vision deficiencies; a scorer keeps
its colour in every chart.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ASSETS = Path("docs/assets")
DPI = 200
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    neutral: str
    series: tuple[str, ...]


THEMES = {
    "light": Theme(
        name="light",
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        neutral="#a8a79f",
        series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    ),
    "dark": Theme(
        name="dark",
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        neutral="#64635e",
        series=("#3987e5", "#d95926", "#199e70", "#c98500"),
    ),
}

DATASETS = {"squad_v2": "SQuAD v2", "musique": "MuSiQue", "hotpotqa": "HotpotQA"}
TARGETS = {"squad_v2": 0.2, "musique": 0.5, "hotpotqa": 0.5}  # error target per dataset
SCORERS = {
    "estimator": ("Sufficiency estimator", 0),
    "hand-weighted": ("Hand-weighted blend", 1),
    "reader margin": ("Reader margin alone", 2),
}
BAR_LABELS = {
    "estimator": "Sufficiency estimator",
    "estimator (no reader)": "Estimator without reader",
    "reader margin": "Reader margin alone",
    "hand-weighted": "Hand-weighted blend",
    "retrieval score": "Retrieval score alone",
}


def use_theme(mode: str) -> Theme:
    theme = THEMES[mode]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink2,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return theme


def _title(fig: Figure, title: str, subtitle: str, theme: Theme) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, weight="bold")
    fig.text(
        0.012,
        0.975 - 0.32 / fig.get_figheight(),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=theme.ink2,
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)
    return path


def _percent_axis(ax: Axes, axis: str, ticks: tuple[float, ...]) -> None:
    labels = [f"{t:.0%}" for t in ticks]
    if axis == "x":
        ax.set_xticks(ticks, labels)
    else:
        ax.set_yticks(ticks, labels)


# ------------------------------------------------------------------------ banner


def banner(mode: str) -> str:
    t = THEMES[mode]
    width, height = 1200, 260
    blue, orange, green = t.series[0], t.series[1], t.series[2]
    x0, x1, y = 600, 1130, 130
    shapes = [
        f'<rect x="{x0}" y="{y - 7}" width="{x1 - x0}" height="14" rx="7" fill="{t.grid}"/>',
        f'<rect x="{x0}" y="{y - 7}" width="{(x1 - x0) * 0.38}" height="14" rx="7" '
        f'fill="{orange}" fill-opacity="0.85"/>',
        f'<rect x="{x0 + (x1 - x0) * 0.38}" y="{y - 7}" width="{(x1 - x0) * 0.27}" '
        f'height="14" fill="{blue}" fill-opacity="0.35"/>',
        f'<rect x="{x0 + (x1 - x0) * 0.65}" y="{y - 7}" width="{(x1 - x0) * 0.35}" '
        f'height="14" rx="7" fill="{green}" fill-opacity="0.9"/>',
        f'<line x1="{x0 + (x1 - x0) * 0.65}" y1="{y - 30}" x2="{x0 + (x1 - x0) * 0.65}" '
        f'y2="{y + 30}" stroke="{t.ink}" stroke-width="2" stroke-dasharray="5 4"/>',
        f'<text x="{x0 + (x1 - x0) * 0.65}" y="{y - 40}" font-family="{FONT}" font-size="14" '
        f'text-anchor="middle" fill="{t.ink2}">guaranteed threshold</text>',
    ]
    labels = [
        (0.19, "abstain", orange),
        (0.515, "fix the context", blue),
        (0.825, "answer", green),
    ]
    for frac, text, color in labels:
        cx = x0 + (x1 - x0) * frac
        shapes.append(f'<circle cx="{cx}" cy="{y + 48}" r="6" fill="{color}"/>')
        shapes.append(
            f'<text x="{cx + 12}" y="{y + 53}" font-family="{FONT}" font-size="16" '
            f'fill="{t.ink}">{text}</text>'
        )
    shapes.append(
        f'<text x="{x0}" y="{y + 92}" font-family="{FONT}" font-size="14" fill="{t.ink2}">'
        f"P(the retrieved passages are enough)  0 → 1</text>"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Sufficiency: know when retrieved context is enough to answer, fix it cheaply when it is not, and abstain with a guaranteed error rate">
  <rect width="{width}" height="{height}" rx="18" fill="{t.surface}"/>
  <text x="56" y="118" font-family="{FONT}" font-size="64" font-weight="700" fill="{t.ink}">Sufficiency</text>
  <text x="58" y="162" font-family="{FONT}" font-size="22" fill="{t.ink2}">Is the retrieved context enough to answer?</text>
  <text x="58" y="192" font-family="{FONT}" font-size="22" fill="{t.ink2}">Know it, fix it cheaply, or abstain.</text>
  {"".join(shapes)}
</svg>
"""


# ----------------------------------------------------------------------- charts


def plot_discrimination(summaries: list[dict[str, Any]], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    order = list(BAR_LABELS)
    fig, axes = plt.subplots(1, len(summaries), figsize=(7.6, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    sizes = {s["test"] for s in summaries}
    test_sizes = (
        f"{sizes.pop():,} per dataset"
        if len(sizes) == 1
        else ", ".join(f"{s['test']:,}" for s in summaries)
    )
    _title(
        fig,
        "Is the context enough? Which signal knows",
        f"Area under the ROC curve on held-out questions ({test_sizes}); top 5 passages "
        "by BM25. 50% = chance.",
        theme,
    )
    for ax, s in zip(axes, summaries, strict=True):
        by = {r["scorer"]: r["auroc"] for r in s["results"]}
        names = [n for n in order if n in by][::-1]
        for y, name in enumerate(names):
            ours = name == "estimator"
            color = theme.series[0] if ours else theme.neutral
            ax.plot([0.5, by[name]], [y, y], color=color, lw=2, solid_capstyle="butt")
            ax.scatter(
                [by[name]], [y], s=70, color=color, zorder=3, edgecolors=theme.surface, linewidths=2
            )
            ax.text(
                max(by[name], 0.5) + 0.02,
                y,
                f"{by[name]:.0%}",
                va="center",
                fontsize=9,
                color=theme.ink if ours else theme.ink2,
                weight="bold" if ours else "normal",
            )
        ax.set_yticks(range(len(names)), [BAR_LABELS[n] for n in names])
        ax.set_xlim(0.45, 1.08)
        _percent_axis(ax, "x", (0.5, 0.75, 1.0))
        ax.set_title(
            f"{DATASETS.get(s['dataset'], s['dataset'])}\n{s['sufficient_share']:.0%} sufficient",
            fontsize=9.5,
            color=theme.ink,
            loc="left",
        )
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=5)
        ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.7, bottom=0.1, wspace=0.12)
    return _save(fig, out)


def reliability(
    probs: np.ndarray, y: np.ndarray, bins: int = 10
) -> tuple[list[float], list[float], list[int]]:
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in itertools.pairwise(edges):
        mask = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if mask.sum() >= 10:
            xs.append(float(probs[mask].mean()))
            ys.append(float(y[mask].mean()))
            ns.append(int(mask.sum()))
    return xs, ys, ns


def plot_calibration(curves: dict[str, dict[str, Any]], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    fig, axes = plt.subplots(1, len(curves), figsize=(7.6, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    _title(
        fig,
        "When it says 80%, is it right 80% of the time?",
        "Predicted chance that the context is enough (x) against the share that really was "
        "(y), held-out questions,\nbins of at least 10. ECE = expected calibration error.",
        theme,
    )
    for ax, (name, c) in zip(axes, curves.items(), strict=True):
        ax.plot([0, 1], [0, 1], color=theme.neutral, lw=1, ls=(0, (3, 2)))
        for key, slot in (("estimator", 0), ("hand", 1)):
            xs, ys, _ = reliability(c[key], c["y"])
            ax.plot(
                xs,
                ys,
                color=theme.series[slot],
                lw=2,
                marker="o",
                ms=4.5,
                markeredgecolor=theme.surface,
                markeredgewidth=1.2,
            )
        ax.text(
            0.04,
            0.95,
            f"ECE {c['ece_estimator']:.3f}",
            color=theme.ink,
            fontsize=9,
            va="top",
            weight="bold",
        )
        ax.text(0.1, 0.85, f"ECE {c['ece_hand']:.3f}", color=theme.ink2, fontsize=9, va="top")
        ax.scatter([0.06], [0.925], s=18, color=theme.series[0])
        ax.scatter([0.06], [0.825], s=18, color=theme.series[1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _percent_axis(ax, "x", (0.0, 0.5, 1.0))
        _percent_axis(ax, "y", (0.0, 0.5, 1.0))
        ax.set_title(DATASETS.get(name, name), fontsize=9.5, color=theme.ink, loc="left")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)
        ax.set_aspect("equal")
    handles = [Line2D([0], [0], color=theme.series[s], lw=2, marker="o", ms=4.5) for s in (0, 1)]
    handles.append(Line2D([0], [0], color=theme.neutral, lw=1, ls=(0, (3, 2))))
    fig.legend(
        handles,
        ["Sufficiency estimator", "Hand-weighted blend", "Perfectly calibrated"],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.15 / 3.6, bottom=0.2, wspace=0.18)
    return _save(fig, out)


def risk_coverage(scores: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    wrong = (~ok[order]).astype(float)
    n = np.arange(1, len(order) + 1)
    return n / len(order), np.cumsum(wrong) / n


def plot_risk_coverage(
    selective: dict[str, dict[str, Any]], results: list[dict[str, Any]], out: Path, mode: str
) -> Path:
    theme = use_theme(mode)
    by_name = {r["dataset"]: r for r in results}
    fig, axes = plt.subplots(1, len(selective), figsize=(7.6, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    _title(
        fig,
        "Answer only when the context is enough",
        "Error rate among answered questions (y) as more questions are answered (x), "
        "highest score first, on\nheld-out questions. Rings: the certified threshold's average "
        "error and coverage (target dashed: 20% or 50%).",
        theme,
    )
    for ax, (name, data) in zip(axes, selective.items(), strict=True):
        ok, half = data["ok"], int(data["half"])
        rows = {r["scorer"]: r for r in by_name[name]["rows"]}
        for scorer, (_, slot) in SCORERS.items():
            cov, risk = risk_coverage(data[scorer][half:], ok[half:])
            start = 19  # skip the first answers, where one error swings the rate wildly
            ax.plot(cov[start:], risk[start:], color=theme.series[slot], lw=2)
            alpha = TARGETS.get(name, 0.2)
            point = (rows[scorer].get("splits") or rows[scorer])[f"ltt@{alpha}"]
            if point["coverage"] > 0:
                ax.scatter(
                    [point["coverage"]],
                    [point["selective_error"]],
                    s=60,
                    facecolor=theme.surface,
                    edgecolor=theme.series[slot],
                    linewidths=2,
                    zorder=4,
                )
        ax.axhline(TARGETS.get(name, 0.2), color=theme.ink, lw=1, ls=(0, (3, 2)))
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1)
        _percent_axis(ax, "x", (0.25, 0.5, 0.75, 1.0))
        _percent_axis(ax, "y", (0.0, 0.2, 0.5, 1.0))
        ax.set_xlabel("answered")
        ax.set_title(
            f"{DATASETS.get(name, name)}\nalways answering: "
            f"{1 - by_name[name]['accuracy_if_always_answering']:.0%} wrong",
            fontsize=9.5,
            color=theme.ink,
            loc="left",
        )
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)
    handles = [Line2D([0], [0], color=theme.series[s], lw=2) for _, s in SCORERS.values()]
    handles.append(Line2D([0], [0], color=theme.ink, lw=1, ls=(0, (3, 2))))
    fig.legend(
        handles,
        [*(label for label, _ in SCORERS.values()), "error target"],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.45 / 3.9, bottom=0.25, wspace=0.16)
    return _save(fig, out)


ACTION_LABELS = {
    "keep": "keep",
    "expand": "expand to top 10",
    "feedback": "query feedback",
    "decompose": "split the question",
    "hop": "follow names",
    "switch": "switch to dense",
    "decompose-llm": "LLM decomposition",
}


def _spread(points: list[tuple[float, float, str]], gap: float) -> list[tuple[float, float, str]]:
    """Nudge label heights apart so labels of nearby points do not overlap."""
    placed: list[tuple[float, float, str]] = []
    for x, y, text in sorted(points, key=lambda p: p[1]):
        for px, py, _ in placed:
            if abs(px - x) < 0.9 and y - py < gap:
                y = py + gap
        placed.append((x, y, text))
    return placed


def plot_recovery(summary: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    rows = {r["policy"]: r for r in summary["rows"]}
    fig = plt.figure(figsize=(7.6, 4.2))
    _title(
        fig,
        "Which fix makes the context enough, and at what cost?",
        f"{summary['test_questions']:,} held-out MuSiQue questions, 2 to 4 hops, open "
        f"retrieval over {summary['corpus']:,} paragraphs. Success = every\nsupporting "
        "paragraph is in the context. Cost = retrieval and model calls per question.",
        theme,
    )
    ax = fig.add_axes((0.08, 0.12, 0.6, 0.6))
    fixed = [p for p in rows if p.startswith("always ") and p != "always decompose-gold"]
    shown = [rows[p] for p in fixed] + [r for p, r in rows.items() if not p.startswith("always")]
    top = max(r["success"] for r in rows.values())
    ax.set_ylim(0, top * 1.18 + 0.02)
    ax.set_xlim(-0.2, max(r["cost"] for r in shown) * 1.08 + 1.6)
    labels = []
    for p in fixed:
        r = rows[p]
        ax.scatter([r["cost"]], [r["success"]], s=46, color=theme.neutral, zorder=3)
        name = p.removeprefix("always ")
        labels.append((r["cost"], r["success"], ACTION_LABELS.get(name, name)))
    span = top * 1.18 + 0.02
    # points crowded around one call per question get their labels in the free space
    # below them, so the learned policy's line stays visible
    crowded = sorted((p for p in labels if 0.5 <= p[0] <= 1.6), key=lambda p: -p[1])
    placed = [(x, y, t, 1.6, span * (0.23 - 0.058 * i)) for i, (x, y, t) in enumerate(crowded)]
    rest = [p for p in labels if p not in crowded]
    spread = _spread(rest, gap=span * 0.075)
    placed += [
        (x, y, t, x + 0.35, ly)
        for (x, y, t), (_, ly, _) in zip(sorted(rest, key=lambda p: p[1]), spread, strict=True)
    ]
    for x, y, text, lx, ly in placed:
        ax.plot([x + 0.04, lx - 0.03], [y, ly], color=theme.axis, lw=0.8, zorder=2)
        ax.text(
            lx,
            ly,
            text,
            fontsize=8.5,
            color=theme.ink2,
            va="center",
            zorder=6,
            bbox={"boxstyle": "square,pad=0.1", "facecolor": theme.surface, "edgecolor": "none"},
        )
    learned = sorted(
        (r for p, r in rows.items() if p.startswith("learned")), key=lambda r: r["cost"]
    )
    ax.plot(
        [r["cost"] for r in learned],
        [r["success"] for r in learned],
        color=theme.series[0],
        lw=2,
        marker="o",
        ms=6,
        zorder=4,
        markeredgecolor=theme.surface,
        markeredgewidth=1.5,
    )
    cascade = rows["rule cascade (hand-weighted)"]
    ax.scatter(
        [cascade["cost"]],
        [cascade["success"]],
        s=70,
        marker="s",
        zorder=4,
        color=theme.series[1],
        edgecolors=theme.surface,
        linewidths=1.5,
    )
    oracle = rows["oracle (deployable actions)"]
    ax.scatter(
        [oracle["cost"]],
        [oracle["success"]],
        s=140,
        marker="*",
        zorder=4,
        color=theme.series[2],
        edgecolors=theme.surface,
        linewidths=1,
    )
    gold = rows.get("always decompose-gold")
    if gold:
        ax.axhline(gold["success"], color=theme.neutral, lw=1, ls=(0, (3, 2)))
        ax.text(
            0.98,
            gold["success"] + 0.008,
            "annotated sub-questions, hop by hop (upper bound)",
            ha="right",
            va="bottom",
            fontsize=8,
            color=theme.muted,
            transform=ax.get_yaxis_transform(),
        )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("calls per question")
    ax.set_ylabel("context sufficient")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)
    handles = [
        Line2D([0], [0], color=theme.series[0], lw=2, marker="o", ms=6),
        Line2D([0], [0], color=theme.series[1], lw=0, marker="s", ms=7),
        Line2D([0], [0], color=theme.series[2], lw=0, marker="*", ms=11),
        Line2D([0], [0], color=theme.neutral, lw=0, marker="o", ms=6),
    ]
    fig.legend(
        handles,
        [
            "Learned policy\n(cost weight swept)",
            "Rule cascade on\nhand-weighted score",
            "Oracle: cheapest action\nthat worked, in hindsight",
            "One action for\nevery question",
        ],
        loc="center left",
        bbox_to_anchor=(0.71, 0.42),
        labelspacing=1.3,
    )
    return _save(fig, out)


# -------------------------------------------------------------------- animation


def animate_threshold(
    name: str, data: dict[str, Any], rule: dict[str, Any], out: Path, mode: str
) -> Path:
    """Sweep the answer threshold from strict to lenient and stop at the certified one."""
    theme = use_theme(mode)
    ok, half = data["ok"], int(data["half"])
    scores, good = data["estimator"][half:], ok[half:]
    target = rule["threshold"]
    fig = plt.figure(figsize=(7.6, 4.2))
    _title(
        fig,
        f"Choosing when to answer ({DATASETS.get(name, name)})",
        "Each dot is a held-out question, placed by its sufficiency estimate. Questions to "
        "the right of the line are\nanswered. Green: the answer would be right. "
        "Orange: it would be wrong.",
        theme,
    )
    ax = fig.add_axes((0.05, 0.25, 0.6, 0.48))
    side = fig.add_axes((0.72, 0.25, 0.25, 0.48))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(0.05, 0.95, len(scores))
    colors = [theme.series[2] if g else theme.series[1] for g in good]
    ax.scatter(scores, jitter, s=9, c=colors, alpha=0.85, linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    _percent_axis(ax, "x", (0.0, 0.25, 0.5, 0.75, 1.0))
    ax.set_xlabel("estimated chance that the context is enough")
    ax.tick_params(length=0, pad=4)
    line = ax.axvline(1.0, color=theme.ink, lw=1.6)
    shade = ax.axvspan(0, 1.0, color=theme.neutral, alpha=0.18, lw=0)
    cov, risk = risk_coverage(scores, good)
    side.plot(cov, risk, color=theme.series[0], lw=2)
    side.axhline(rule["alpha"], color=theme.ink, lw=1, ls=(0, (3, 2)))
    side.text(
        1.0,
        rule["alpha"] + 0.02,
        f"{rule['alpha']:.0%} target",
        ha="right",
        fontsize=8,
        color=theme.ink2,
    )
    dot = side.scatter(
        [0], [0], s=55, color=theme.series[0], edgecolors=theme.surface, linewidths=1.5, zorder=4
    )
    side.set_xlim(0, 1)
    side.set_ylim(0, max(0.6, float(risk.max()) + 0.05))
    _percent_axis(side, "x", (0.0, 0.5, 1.0))
    side.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    side.set_xlabel("answered")
    side.set_title("error among answered", fontsize=9, color=theme.ink2, loc="left")
    side.grid(True)
    side.tick_params(length=0, pad=4)
    status = fig.text(0.05, 0.035, "", fontsize=10.5, color=theme.ink)
    sweep = list(np.linspace(1.0, target, 40)) if np.isfinite(target) else []
    frames = [*sweep, *([target] * 18)] if sweep else [1.0]

    def draw(i: int) -> list[Any]:
        t = float(frames[i])
        line.set_xdata([t, t])
        shade.set_x(0)
        shade.set_width(t)
        answered = scores >= t
        n = int(answered.sum())
        err = float((~good[answered]).mean()) if n else 0.0
        dot.set_offsets([[n / len(scores), err]])
        final = i >= len(sweep) - 1 and bool(sweep)
        status.set_text(
            f"threshold {t:.2f}: answers {n / len(scores):.0%} of questions, "
            f"{err:.0%} of those wrong"
            + ("   (certified: at most 20% with 90% confidence)" if final else "")
        )
        status.set_fontweight("bold" if final else "normal")
        return [line, shade, dot, status]

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=90, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=11), dpi=110)
    plt.close(fig)
    print(out)
    return out


# --------------------------------------------------------------------- inputs


def calibration_curves(estimate_dir: Path) -> dict[str, dict[str, Any]]:
    from sufficiency.estimator import SufficiencyEstimator
    from sufficiency.features import (
        Featurizer,
        LexicalFeatures,
        ReaderFeatures,
        RetrievalFeatures,
    )
    from sufficiency.metrics import expected_calibration_error

    results = {s["dataset"]: s for s in json.loads((estimate_dir / "results.json").read_text())}
    curves = {}
    for name in DATASETS:
        model = estimate_dir / f"estimator-{name}.joblib"
        if name not in results or not model.exists():
            continue
        featurizer = Featurizer(RetrievalFeatures(), LexicalFeatures(), ReaderFeatures())
        estimator = SufficiencyEstimator(featurizer).load(model)
        cached = np.load(estimate_dir / f"features-{name}.npz")
        probs = estimator.predict_matrix(cached["test_x"])
        hand = np.clip(cached["test_hand"], 0, 1)
        y = cached["test_y"].astype(float)
        curves[name] = {
            "estimator": probs,
            "hand": hand,
            "y": y,
            "ece_estimator": expected_calibration_error(probs, y),
            "ece_hand": expected_calibration_error(hand, y),
        }
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", default="runs/estimate")
    parser.add_argument("--selective", default="runs/selective")
    parser.add_argument("--recover", default="runs/recover/results.json")
    parser.add_argument("--animate", default="squad_v2", help="dataset for the animation")
    parser.add_argument("--animate-from", default="runs/selective", help="selective run to animate")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        (ASSETS / f"banner-{mode}.svg").write_text(banner(mode))
        print(ASSETS / f"banner-{mode}.svg")

    estimate = Path(args.estimate)
    if (estimate / "results.json").exists():
        summaries = json.loads((estimate / "results.json").read_text())
        curves = calibration_curves(estimate)
        for mode in THEMES:
            plot_discrimination(summaries, ASSETS / f"discrimination-{mode}.png", mode)
            plot_calibration(curves, ASSETS / f"calibration-{mode}.png", mode)

    selective = Path(args.selective)
    if (selective / "results.json").exists():
        results = json.loads((selective / "results.json").read_text())
        scores = {
            name: dict(np.load(selective / f"scores-{name}.npz"))
            for name in DATASETS
            if (selective / f"scores-{name}.npz").exists()
        }
        for mode in THEMES:
            plot_risk_coverage(scores, results, ASSETS / f"risk-coverage-{mode}.png", mode)
        source = Path(args.animate_from) / f"scores-{args.animate}.npz"
        if source.exists():
            data = dict(np.load(source))
            half = int(data["half"])
            from sufficiency.selective import SelectiveRule

            rule = SelectiveRule.fit(
                data["estimator"][:half], ~data["ok"][:half], 0.2, method="ltt", delta=0.1
            )
            for mode in THEMES:
                animate_threshold(
                    args.animate,
                    data,
                    {"threshold": rule.threshold, "alpha": 0.2},
                    ASSETS / f"threshold-{mode}.gif",
                    mode,
                )

    recover = Path(args.recover)
    if recover.exists():
        summary = json.loads(recover.read_text())
        for mode in THEMES:
            plot_recovery(summary, ASSETS / f"recovery-{mode}.png", mode)


if __name__ == "__main__":
    main()
