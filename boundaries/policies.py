"""Gating policies that map a proposed agent action to execute / ask / block.

B0  broad permission        : execute everything (status quo of one-time broad grants)
B1  always confirm          : ask before every action
A   scoped rules            : user-defined allow/ask/deny rules + static risk tier for unruled pairs
B   boundary-aware risk model: calibrated P(approve) conditioned on action AND user boundaries,
                              deny rules as hard constraints, uncertainty band -> ask
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from .sim import CATEGORIES, RECIPIENTS, SCOPES, TOOLS

EXECUTE, ASK, BLOCK = 0, 1, 2

ACTION_COLS = ["tool", "category", "recipient", "scope"]
RULES = ["allow", "ask", "deny", "none"]


# ---------------------------------------------------------------- features
def _onehot(df, col, values):
    return pd.DataFrame({f"{col}={v}": (df[col] == v).astype(int) for v in values}, index=df.index)


def user_rule_profile(rules):
    """Per-user summary of their stated boundaries (how restrictive they are overall,
    per tool and per category). Lets the model generalise from stated rules to unruled pairs."""
    prof = {}
    for uid, r in rules.items():
        vals = list(r.values())
        n = max(len(vals), 1)
        p = {"u_deny_frac": vals.count("deny") / n, "u_allow_frac": vals.count("allow") / n}
        for t in TOOLS:
            tv = [v for (tt, _), v in r.items() if tt == t]
            p[f"tool_deny_{t}"] = tv.count("deny") / len(tv) if tv else np.nan
        for c in CATEGORIES:
            cv = [v for (_, cc), v in r.items() if cc == c]
            p[f"cat_deny_{c}"] = cv.count("deny") / len(cv) if cv else np.nan
        prof[uid] = p
    return pd.DataFrame.from_dict(prof, orient="index")


def build_features(df, rules, use_boundaries=True):
    parts = [
        _onehot(df, "tool", TOOLS), _onehot(df, "category", CATEGORIES),
        _onehot(df, "recipient", RECIPIENTS), _onehot(df, "scope", SCOPES),
        df[["reversible", "external", "relevance"]],
    ]
    if use_boundaries:
        parts.append(_onehot(df, "rule", RULES))
        prof = user_rule_profile(rules).loc[df["user_id"]].reset_index(drop=True)
        prof.index = df.index
        # only keep the profile entries relevant to this action's tool and category
        same_tool = np.array([prof.at[i, f"tool_deny_{t}"] for i, t in zip(df.index, df["tool"])])
        same_cat = np.array([prof.at[i, f"cat_deny_{c}"] for i, c in zip(df.index, df["category"])])
        parts.append(pd.DataFrame({
            "u_deny_frac": prof["u_deny_frac"], "u_allow_frac": prof["u_allow_frac"],
            "same_tool_deny": same_tool, "same_cat_deny": same_cat,
        }, index=df.index))
    return pd.concat(parts, axis=1).astype(float)


# ---------------------------------------------------------------- simulated confirmation
def resolve(decisions, y, ask_acc0, habituation, rng):
    """Outcome of each decision. An ASK is answered by the simulated user, whose accuracy
    drops with the share of actions they are asked about (confirmation fatigue).
    Returns executed (bool array) and the user accuracy used."""
    asked = decisions == ASK
    acc = max(ask_acc0 - habituation * asked.mean(), 0.5)
    correct = rng.random(len(y)) < acc
    user_says_yes = np.where(correct, y == 1, y == 0)
    executed = (decisions == EXECUTE) | (asked & user_says_yes)
    return executed, acc


# ---------------------------------------------------------------- policies
def policy_broad(df):
    return np.full(len(df), EXECUTE)


def policy_always_ask(df):
    return np.full(len(df), ASK)


def static_tier(df):
    """Generic risk tier used when no user rule applies: confirm irreversible or external actions."""
    risky = (df["reversible"] == 0) | (df["external"] == 1)
    return np.where(risky, ASK, EXECUTE)


def policy_rules(df, with_ask=True):
    """Design A. Explicit user rules first; static risk tier for unruled pairs.
    with_ask=False is the allow/deny-list ablation (ask rules and fallback both execute)."""
    out = static_tier(df) if with_ask else np.full(len(df), EXECUTE)
    out = np.where(df["rule"] == "allow", EXECUTE, out)
    out = np.where(df["rule"] == "ask", ASK if with_ask else EXECUTE, out)
    out = np.where(df["rule"] == "deny", BLOCK, out)
    return out
