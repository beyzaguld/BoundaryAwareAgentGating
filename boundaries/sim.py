"""Synthetic scenario generator for agent actions, user preferences and user-defined boundaries.

Every simulated user has a latent preference function that decides whether they would
approve a proposed agent action (ground truth). Users also write an explicit, coarse and
incomplete set of boundary rules over (tool, data category) pairs. The agent only sees the
action description and the stated rules, never the latent preferences.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# tool -> (reversible, external, base risk)
TOOLS = {
    "read_file":    (True,  False, 0.0),
    "create_event": (True,  False, 0.2),
    "write_file":   (True,  False, 0.4),
    "delete_event": (False, False, 0.8),
    "delete_file":  (False, False, 1.2),
    "share_file":   (True,  True,  1.0),
    "send_email":   (False, True,  1.0),
    "make_payment": (False, True,  2.0),
}
TOOL_FREQ = np.array([0.24, 0.14, 0.14, 0.07, 0.08, 0.10, 0.17, 0.06])

CATEGORIES = {"public": 0.0, "work": 0.3, "personal": 0.8, "health": 1.5, "finance": 1.5}
CAT_FREQ = np.array([0.15, 0.40, 0.25, 0.08, 0.12])

RECIPIENTS = {"none": 0.0, "self": 0.0, "team": 0.3, "external_known": 0.8, "external_unknown": 1.8}
EXTERNAL_RECIP = ["self", "team", "external_known", "external_unknown"]
EXTERNAL_RECIP_FREQ = np.array([0.15, 0.40, 0.35, 0.10])

SCOPES = {"single": 0.0, "few": 0.3, "bulk": 1.0}
SCOPE_FREQ = np.array([0.70, 0.22, 0.08])

ARCHETYPES = {
    # name: (base permissiveness, sensitivity scale, population share)
    "conservative": (2.0, 1.4, 0.30),
    "balanced":     (3.0, 1.0, 0.45),
    "delegator":    (4.0, 0.7, 0.25),
}

MISTAKE_TYPES = ["off_task", "wrong_recipient", "overbroad_scope"]
RULE_MARGIN = 1.0          # |typical score| above which a user writes allow/deny instead of ask
RULE_FLIP_PROB = 0.05      # probability that a stated rule is misspecified


@dataclass
class SimConfig:
    n_users: int = 300
    actions_per_user: int = 60
    rule_coverage: float = 0.6   # share of (tool, category) pairs for which a user writes a rule
    mistake_rate: float = 0.15   # share of agent proposals that contain an injected agent error
