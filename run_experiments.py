"""Main experiment: baselines, Design A, Design B and mechanism-check ablations.

Usage:  python run_experiments.py [--seeds 10] [--out results]
Writes per-seed metrics, summary tables with 95% CIs, subgroup tables, a coverage sweep,
a habituation sensitivity analysis and case-level examples to the output folder.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundaries.metrics import evaluate, expected_cost_fn, mean_ci, prob_metrics
from boundaries.policies import (BoundaryRiskModel, policy_always_ask, policy_broad,
                                 policy_rules, EXECUTE, ASK, BLOCK)
from boundaries.sim import SimConfig, describe, generate

SPLIT = (0.6, 0.2, 0.2)  # train / validation / test, split by USER (test users are unseen)


def split_users(df, seed):
    users = np.random.default_rng(seed + 999).permutation(df["user_id"].unique())
    n = len(users)
    a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    pick = lambda u: df[df["user_id"].isin(u)].reset_index(drop=True)
    return pick(users[:a]), pick(users[a:b]), pick(users[b:])


def fit_models(train, val, rules, seed):
    cost_fn = expected_cost_fn()
    m = {
        "B_full": BoundaryRiskModel(True, True, seed).fit(train, val, rules).tune_thresholds(val, rules, cost_fn),
        "B_no_boundaries": BoundaryRiskModel(False, False, seed).fit(train, val, rules).tune_thresholds(val, rules, cost_fn),
        "B_soft_deny": BoundaryRiskModel(True, False, seed).fit(train, val, rules).tune_thresholds(val, rules, cost_fn),
    }
    nb = BoundaryRiskModel(True, True, seed)
    nb.raw, nb.cal = m["B_full"].raw, m["B_full"].cal   # same model, only the ask band removed
    m["B_no_ask_band"] = nb.tune_thresholds(val, rules, cost_fn, band=False)
    return m
