"""matched_pairs.py — PLAN_AMENDMENT_3 §C/§D/§F frozen production module (post-C1),
THINNED under PLAN_AMENDMENT_4 §H (GATE-1 code stage, 2026-07-12).

v4 thinning: the construction layer (CertificateError, fingerprint,
BACKBONE_PRESET, DRAW_SEEDS, make_arms) now DELEGATES to the single-owner
shared constructor matched_construction.py — byte-compatible with the pristine
v3 path (matched_feasibility.build_matched_pair) via matched_construction's
certificate (xvii) and the golden fixtures in test_matched_construction.py.
Everything below the construction layer (§B-4 panel lists, §D invocation
gates, §F typed/transactional retries, §G budget cap) is UNCHANGED v3 text.

Single source of truth for the displacement-matched contrast:
  - the frozen §B-4 panel (judgeable / report-only / unconstructible lists),
  - frozen draw-seed sets, §F retry rule (typed technical failures only,
    MAX_ATTEMPTS enforced cumulatively across process restarts via the
    append-only ATTEMPTS ledger), transactional per-cell row emission,
  - §D invocation gates incl. smoke whitelist + run-id isolation,
  - §G budget accounting with hard cap.

C1 code-review fixes: fail-closed certificates, smoke/output
isolation, typed+transactional retries, budget ledger, -O refusal.
"""
import json
import math
import os
import re
import time

import numpy as np

from matched_construction import (  # v4 §H thinning — single construction owner
    BACKBONE_PRESET,
    CertificateError,
    DRAW_SEEDS,
    fingerprint,
    legacy_arms,
)

if not __debug__:  # §C-6 certificates rely on assert — refuse optimizer mode
    raise SystemExit("matched_pairs: refusing to run under python -O "
                     "(certificate asserts would be disabled)")

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PILOT_DIR, "..", ".."))


# ------------------------------------------------------------ frozen constants ----

MAX_ATTEMPTS = 3          # §F: initial + at most 2 identical-config technical retries
BUDGET_GPU_H_CAP = 25.0   # §G hard cap (matched runs incl. retries + smoke)

# §B-4 frozen panel (R3 exposure rule; runs/MATCHED_FEASIBILITY/feasibility_r3.json)
GRAPHS = {
    "ultra": ["fb-v1", "fb-v2", "fb-v3", "fb-v4", "nl-50", "nl-75", "nl-100", "wk-75"],
    "trix": ["fb-v1", "fb-v2", "fb-v3", "nl-50", "nl-75", "wk-75"],  # constructible only
    "flock": ["nl-50", "nl-75", "nl-100", "wk-75"],
}
JUDGEABLE = {
    "ultra": {"fb-v1", "fb-v2", "fb-v3", "fb-v4", "nl-50", "nl-75", "nl-100", "wk-75"},
    "trix": set(),                       # entire tier report-only (low exposure)
    "flock": {"nl-50", "nl-75", "nl-100", "wk-75"},
}
UNCONSTRUCTIBLE = {"ultra": set(), "trix": {"fb-v4", "nl-100"}, "flock": set()}
LOW_QUERY_MASS_FLAG = {"ultra": {"fb-v3"}, "trix": set(), "flock": set()}
OOF_GRAPHS = {"nl-50", "nl-75", "nl-100", "wk-75"}  # one-shot-sensitive (3rd reuse)

# §C-9 frozen smoke panel — smoke may run EXACTLY these graphs, nothing else
SMOKE_GRAPHS = {"ultra": ["fb-v1"], "trix": ["fb-v1"], "flock": ["nl-25"]}
SMOKE_RUN_PREFIX = "MATCHED_SMOKE_"
PROD_RUN_IDS = {"ultra": "MATCHED_ULTRA_prereg_v3",
                "trix": "MATCHED_TRIX_prereg_v3",
                "flock": "MATCHED_FLOCK_prereg_v3"}

CKPTS = {
    "ultra": ("ultra_3g", "ultra_4g", "ultra_50g"),
    "trix": ("entity_prediction.pth",),
    "flock": ("flock_entity.pth",),
}

ARMS = {
    "ultra": ("Wm", "Xm", "Xfull"),
    "trix": ("Wm", "Xm", "Xfull"),
    "flock": ("Wm", "Xm"),
}
CLAIM_ARMS = ("Wm", "Xm")  # strata (§E-7) computed for these only

# §E frozen decision constants
DELTA_MARGIN = 0.30
DELTA_MARGIN_GRID = (0.20, 0.25, 0.30, 0.35, 0.40)   # E-5 descriptive grid
XM_COLLAPSE_LINE = 0.50
WM_BAND = (0.97, 1.03)
FLOCK_CONSISTENT_MIN = 3     # of 4 judgeable graphs
DOMAIN_OF = {"fb-v1": "FB15k-237", "fb-v2": "FB15k-237", "fb-v3": "FB15k-237",
             "fb-v4": "FB15k-237", "nl-50": "NELL-995", "nl-75": "NELL-995",
             "nl-100": "NELL-995", "wk-75": "Wikidata"}

RUN_TAG = "MATCHED_prereg_v3"
SMOKE_LABEL = "EXPLORATORY_SMOKE_NONCLAIM"
PROD_LABEL = {
    "ultra": "claim-set MATCHED (PLAN_AMENDMENT_3, prereg-v3)",
    "flock": "claim-set MATCHED (PLAN_AMENDMENT_3, prereg-v3)",
    "trix": ("REPORT_ONLY_LOW_EXPOSURE (PLAN_AMENDMENT_3 §B-4; "
             "never outcome-bearing)"),
}
CI_NOTE = ("percentile bootstrap over paired draws, B=10000, seed 0 — sampling "
           "uncertainty CONDITIONAL on the frozen draw generator (§E-4)")


# ---------------------------------------------------------------- construction ----

def make_arms(prof, R_base, draw_seed, backbone):
    """One draw's certified arms — v4 thin delegation to the shared constructor
    (byte-compatible with the pristine v3 path; §C certificates run inside
    matched_construction; failures surface as CertificateError, not retried)."""
    return legacy_arms(prof, R_base, draw_seed, backbone)


def pi_for_arm(arms, arm):
    return {"Wm": arms["pi_w"], "Xm": arms["pi_x"], "Xfull": arms["pi_xfull"]}[arm]


def check_mrr(x, where):
    """§C-6 finite-range certificate: MRR must be finite and in [0,1]."""
    if x is None or not math.isfinite(float(x)) or not (0.0 <= float(x) <= 1.0):
        raise CertificateError(f"{where}: MRR {x!r} not finite in [0,1]")
    return float(x)


def lifted_mask(M, R_base):
    """Boolean mask over 2R for M ∪ (M+R) (strata + exposure helper)."""
    import torch
    m = torch.zeros(2 * R_base, dtype=torch.bool)
    if M:
        idx = torch.tensor(M, dtype=torch.long)
        m[idx] = True
        m[idx + R_base] = True
    return m


# ------------------------------------------------------------------ validation ----

def validate_matched_invocation(backbone, req_graphs, draws, ckpts, arms,
                                limit, smoke, run_id):
    """Frozen §D invocation gate. Returns 'production' or 'smoke'.

    production ⇔ exact frozen graph/draw/ckpt/arm sets, no --limit, no --smoke,
                 run_id == frozen production id.
    smoke      ⇔ --smoke AND graphs == frozen §C-9 smoke panel (exactly) AND
                 draws a nonempty prefix [0..k-1], k<=3 AND run_id startswith
                 MATCHED_SMOKE_ (output isolation).
    Anything else: SystemExit."""
    req = list(req_graphs)
    if any(not g for g in req):  # empty tokens are refused, never filtered (C3)
        raise SystemExit(f"empty graph token in {req!r} (§D)")
    problems = []
    if smoke:
        if sorted(req) != sorted(SMOKE_GRAPHS[backbone]):
            raise SystemExit(f"smoke graphs {sorted(req)} != frozen smoke panel "
                             f"{SMOKE_GRAPHS[backbone]} (§C-9)")
        d = list(draws)
        if not d or d != list(range(len(d))) or len(d) > 3:
            raise SystemExit("smoke draws must be a nonempty prefix [0..k-1], "
                             "k<=3 (§C-9)")
        if not str(run_id).startswith(SMOKE_RUN_PREFIX):
            raise SystemExit(f"smoke run-id must start with {SMOKE_RUN_PREFIX!r} "
                             "(output isolation; §D)")
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(run_id)):
            raise SystemExit("run-id must be a single safe path component "
                             "(no separators/traversal; §D)")
        return "smoke"
    if sorted(req) != sorted(GRAPHS[backbone]):
        problems.append(f"graphs {sorted(req)} != frozen {sorted(GRAPHS[backbone])}")
    if list(draws) != DRAW_SEEDS[backbone]:
        problems.append(f"draw seeds != frozen {DRAW_SEEDS[backbone]}")
    if tuple(ckpts) != CKPTS[backbone]:
        problems.append(f"ckpts {tuple(ckpts)} != frozen {CKPTS[backbone]}")
    if tuple(arms) != ARMS[backbone]:
        problems.append(f"arms {tuple(arms)} != frozen {ARMS[backbone]}")
    if limit:
        problems.append("--limit forbidden in production claim mode")
    if str(run_id) != PROD_RUN_IDS[backbone]:
        problems.append(f"run-id {run_id!r} != frozen {PROD_RUN_IDS[backbone]!r}")
    if str(run_id).startswith(SMOKE_RUN_PREFIX):
        problems.append("production run-id must not use the smoke prefix")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(run_id)):
        problems.append("run-id must be a single safe path component")
    if problems:
        raise SystemExit("matched claim-mode invocation violates "
                         "PLAN_AMENDMENT_3 §D (frozen): " + "; ".join(problems))
    return "production"


def validator_selftest(backbone):
    """Every weakened invocation must be refused (§D selftest, pre-tag gate)."""
    base = dict(backbone=backbone, req_graphs=GRAPHS[backbone],
                draws=DRAW_SEEDS[backbone], ckpts=CKPTS[backbone],
                arms=ARMS[backbone], limit=0, smoke=False,
                run_id=PROD_RUN_IDS[backbone])
    assert validate_matched_invocation(**base) == "production"
    smoke_ok = dict(base, smoke=True, req_graphs=SMOKE_GRAPHS[backbone],
                    draws=[0, 1, 2], run_id=SMOKE_RUN_PREFIX + backbone)
    assert validate_matched_invocation(**smoke_ok) == "smoke"
    weakenings = [
        dict(base, req_graphs=GRAPHS[backbone][:-1]),
        dict(base, draws=DRAW_SEEDS[backbone][:-1]),
        dict(base, ckpts=CKPTS[backbone][:-1] or ("wrong",)),
        dict(base, arms=ARMS[backbone][:-1] or ("Wm",)),
        dict(base, limit=200),
        dict(base, run_id="MATCHED_other"),
        dict(base, run_id=SMOKE_RUN_PREFIX + "sneaky"),
        dict(smoke_ok, req_graphs=["nl-50"]),                 # smoke on OOF
        dict(smoke_ok, req_graphs=GRAPHS[backbone][:1] + ["extra"]),
        dict(smoke_ok, draws=[0, 1, 2, 3]),                   # too many draws
        dict(smoke_ok, draws=[1, 2]),                         # non-prefix draws
        dict(smoke_ok, draws=[]),                             # empty draws
        dict(smoke_ok, run_id=PROD_RUN_IDS[backbone]),        # smoke into prod dir
        dict(smoke_ok, run_id=SMOKE_RUN_PREFIX + "/../" +
             PROD_RUN_IDS[backbone]),                         # path traversal
        dict(smoke_ok, req_graphs=[g for b, gs in SMOKE_GRAPHS.items()
                                   if b != backbone for g in gs
                                   if g not in SMOKE_GRAPHS[backbone]][:1]),
    ]
    refused = 0
    for w in weakenings:
        try:
            validate_matched_invocation(**w)
        except SystemExit:
            refused += 1
    assert refused == len(weakenings), (
        f"{backbone}: only {refused}/{len(weakenings)} weakened invocations refused")
    return (f"{backbone}: selftest PASS ({len(weakenings)} weakened refused, "
            f"production + smoke accepted)")


# ------------------------------------------------- attempts / retries / budget ----

def _ledger_path(out_dir):
    return os.path.join(out_dir, "ATTEMPTS.jsonl")


def log_attempt(out_dir, record):
    os.makedirs(out_dir, exist_ok=True)
    with open(_ledger_path(out_dir), "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            **record}) + "\n")


def ledger_state(out_dir, cell_key):
    """Cumulative ledger view for a cell — survives process restarts.
    Returns dict(starts=int, terminal=bool, success=bool)."""
    p = _ledger_path(out_dir)
    starts, terminal, success = 0, False, False
    if os.path.exists(p):
        with open(p) as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    # C3-8 (v4 tightening): a malformed attempts row could
                    # hide starts and undercount lifetime retries
                    raise SystemExit(f"ATTEMPTS ledger malformed at "
                                     f"{p}:{ln}: {e} (fail-closed; §F)")
                if not isinstance(r, dict):
                    raise SystemExit(f"ATTEMPTS ledger non-object row "
                                     f"{p}:{ln} (fail-closed; §F)")
                if r.get("cell") != cell_key:
                    continue
                ev = r.get("event")
                if ev == "start":
                    starts += 1
                elif ev in ("terminal", "exhausted"):
                    terminal = True
                elif ev == "success":
                    success = True
    return {"starts": starts, "terminal": terminal, "success": success}


def is_technical_failure(e):
    """§F: retry ONLY typed technical failures (crash/OOM/IO). Certificate and
    programming errors are terminal immediately."""
    if isinstance(e, (CertificateError, AssertionError)):
        return False
    if isinstance(e, (OSError, IOError, MemoryError)):
        return True
    if isinstance(e, RuntimeError):
        s = str(e).lower()
        return any(k in s for k in ("out of memory", "cuda", "cudnn", "nccl",
                                    "device-side assert"))
    return False


def run_cell_with_retries(out_dir, cell_key, fn, jsonl_path, error_row_base):
    """§F transactional retry wrapper.

    fn() -> (rows, result). Rows are buffered by fn and written HERE, in one
    append batch, ONLY after fn succeeds — a failed attempt leaves no data rows
    (no partial/duplicate contamination). Attempt starts are logged BEFORE fn
    and counted cumulatively across restarts (lifetime MAX_ATTEMPTS).
    Terminal failure appends exactly one ERROR row."""
    state = ledger_state(out_dir, cell_key)
    if state["terminal"]:  # persisted terminal state — do NOT rerun or re-emit
        return False, "cell already terminal in ATTEMPTS ledger (§F)"
    if state["success"]:
        return False, ("cell already succeeded per ATTEMPTS ledger — rerun "
                       "refused (§F); use --resume cell skipping")
    while True:
        state = ledger_state(out_dir, cell_key)
        if state["starts"] >= MAX_ATTEMPTS:
            err = (f"attempt limit reached ({state['starts']}/{MAX_ATTEMPTS}, "
                   "incl. prior runs)")
            log_attempt(out_dir, {"cell": cell_key, "event": "exhausted",
                                  "attempts_used": state["starts"]})
            append_jsonl(jsonl_path, {**error_row_base, "arm": "ERROR",
                                      "draw": None, "error": err})
            return False, err
        log_attempt(out_dir, {"cell": cell_key, "event": "start",
                              "attempt": state["starts"] + 1})
        try:
            rows, result = fn()
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — classified below
            technical = is_technical_failure(e)
            log_attempt(out_dir, {"cell": cell_key,
                                  "event": ("error" if technical else "terminal"),
                                  "attempt": state["starts"] + 1,
                                  "technical": bool(technical),
                                  "error": repr(e)[:500]})
            if technical:
                continue  # identical-config retry, lifetime budget via ledger
            append_jsonl(jsonl_path, {**error_row_base, "arm": "ERROR",
                                      "draw": None, "error": repr(e)[:500],
                                      "terminal_nontechnical": True})
            return False, repr(e)[:500]
        buf = "".join(json.dumps(r) + "\n" for r in rows)
        try:
            with open(jsonl_path, "a") as f:  # one transactional append batch
                f.write(buf)
        except Exception as e:  # partial write is unrecoverable by retry
            log_attempt(out_dir, {"cell": cell_key, "event": "write_error",
                                  "error": repr(e)[:300]})
            raise SystemExit(f"row batch write failed for {cell_key}: {e!r} — "
                             "manual intervention required (§F)")
        log_attempt(out_dir, {"cell": cell_key, "event": "success",
                              "attempt": state["starts"] + 1, "rows": len(rows)})
        return True, result


GPU_LEDGER = os.path.join(PILOT_DIR, "runs", "gpu_hours_ledger.jsonl")


def log_gpu(cell_key, seconds, mode):
    with open(GPU_LEDGER, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "run": RUN_TAG, "mode": mode, "cell": cell_key,
                            "gpu_sec": round(float(seconds), 1)}) + "\n")


def budget_check():
    """§G hard cap over ALL matched entries (production + smoke + retries)."""
    total = 0.0
    if os.path.exists(GPU_LEDGER):
        with open(GPU_LEDGER) as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    # C3-8 (v4 tightening): malformed rows could undercount
                    # spent budget — fail closed instead of skipping
                    raise SystemExit(f"GPU ledger malformed at "
                                     f"{GPU_LEDGER}:{ln}: {e} (§G)")
                if not isinstance(r, dict):
                    raise SystemExit(f"GPU ledger non-object row "
                                     f"{GPU_LEDGER}:{ln} (§G)")
                if r.get("run") == RUN_TAG:
                    raw = r.get("gpu_sec")
                    if isinstance(raw, bool) or not isinstance(
                            raw, (int, float)):
                        raise SystemExit(f"GPU ledger missing/non-numeric "
                                         f"gpu_sec at {GPU_LEDGER}:{ln} "
                                         "(§G; C4-5)")
                    sec = float(raw)
                    if not math.isfinite(sec) or sec < 0:
                        raise SystemExit(f"GPU ledger invalid gpu_sec at "
                                         f"{GPU_LEDGER}:{ln} (§G)")
                    total += sec
    if total > BUDGET_GPU_H_CAP * 3600:
        raise SystemExit(f"§G budget cap exceeded: {total/3600:.2f} GPU-h > "
                         f"{BUDGET_GPU_H_CAP} — stopping (disclose per §G)")
    return total / 3600


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def refuse_or_resume(out_path, resume):
    """Rerun refusal (§F, both modes since C3): an existing results jsonl
    requires --resume (which skips only exactly-complete cells)."""
    if os.path.exists(out_path) and not resume:
        raise SystemExit(f"{out_path} exists — rerun refused; pass --resume "
                         "to continue only missing cells (§F)")
