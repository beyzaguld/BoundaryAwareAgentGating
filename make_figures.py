"""Figures, paired mechanism-check statistics and LaTeX tables from results/.

Usage:  python make_figures.py [--res results] [--fig ../report/figures] [--tab ../report/tables]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from boundaries.metrics import mean_ci

LABELS = {
    "B0_broad": "B0 Broad permission", "B1_always_ask": "B1 Always confirm",
    "A_rules": "A Scoped rules", "A_no_ask_tier": "A $-$ ask tier",
    "B_full": "B Boundary-aware model", "B_no_boundaries": "B $-$ boundaries",
    "B_no_ask_band": "B $-$ ask band", "B_soft_deny": "B $-$ hard deny",
}
MAIN = ["B0_broad", "B1_always_ask", "A_rules", "B_full"]
ABL = ["A_no_ask_tier", "B_no_boundaries", "B_no_ask_band", "B_soft_deny"]
COLORS = {"B0_broad": "#9e9e9e", "B1_always_ask": "#616161", "A_rules": "#e08a2c", "B_full": "#2b6cb0",
          "A_no_ask_tier": "#f2c08f", "B_no_boundaries": "#90b4dd", "B_no_ask_band": "#5a8fcf", "B_soft_deny": "#1d4577"}


def pipeline_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 46); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec="#333333", lw=1.2, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, c="#333333", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2, linestyle=ls))

    hc = "#dbe8f7"
    box(1, 30, 15, 9, "User task\n(goal in\nnatural language)")
    box(21, 30, 16, 9, "Agent planner\n(simulated)\nproposes tool call")
    box(1, 12, 15, 11, "User-defined\nboundaries\nallow / ask / deny\nper (tool, data)", fc=hc, ec="#2b6cb0", lw=1.8, bold=True)
    box(42, 30, 17, 9, "Action features\ntool, data category,\nrecipient, scope,\nreversible, relevance")
    box(42, 12, 17, 11, "Boundary features\nrule for this pair,\nuser deny/allow share,\nsame-tool / same-data", fc=hc, ec="#2b6cb0", lw=1.8)
    box(63, 21, 15, 12, "Risk model\nGBDT + isotonic\ncalibration\n$\\hat p$ = P(approve)", fc=hc, ec="#2b6cb0", lw=1.8, bold=True)
    box(82, 30, 16, 9, "Gate\n$\\hat p\\geq\\tau_{hi}$: execute\n$\\tau_{lo}<\\hat p<\\tau_{hi}$: ASK\n$\\hat p\\leq\\tau_{lo}$: block", fc=hc, ec="#2b6cb0", lw=1.8)
    box(82, 12, 16, 11, "Hard constraints\ndeny rule $\\to$ block\nallow rule never\nsilently blocked", fc=hc, ec="#2b6cb0", lw=1.8)
    box(62, 1, 36, 7, "Outcome: execute / ask user / block\n$\\to$ UER, FBR, ask rate, expected cost", fs=8)
    box(20, 1, 36, 7, "Design A (rules): stated rule $\\to$ direct lookup;\nunruled pair $\\to$ static tier\n(irreversible or external $\\to$ ask)",
        fc="#fdf0e2", ec="#a35c12", lw=1.4, fs=7.8)
    arrow(16, 34.5, 21, 34.5); arrow(37, 34.5, 42, 34.5); arrow(16, 17.5, 42, 17.5)
    arrow(59, 34.5, 63, 29); arrow(59, 17.5, 63, 25); arrow(78, 27, 82, 33)
    arrow(90, 30, 90, 23); arrow(90, 12, 90, 8.6)
    arrow(10, 12, 20, 6, c="#a35c12", ls="--"); arrow(56, 4.5, 62, 4.5, c="#a35c12", ls="--")
    ax.text(1, 44, "Design B (proposed): boundary-aware calibrated gating", fontsize=10, fontweight="bold", color="#2b6cb0")
    ax.text(99, 44, "blue = human-centred mechanism", fontsize=8, color="#2b6cb0", style="italic", ha="right")
    ax.text(1, 41.2, "Baselines on the same proposals: B0 executes every proposal; B1 asks before every proposal.", fontsize=7.8, color="#555555")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
