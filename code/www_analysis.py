"""Endpoints and decision rules of the WWW 2027 amendment (refine-logs/PREREG_WWW2027.md).

exp1: E1a, the determinism check against the archived matched records and the
      partner-type decomposition of the legacy FB cells; E1b, rules R1, P1 and P2
      under the exposure guard.
exp2: documented predicate merges (rules M1 and M2).
exp3, exp4: the same two experiments on the seven graphs of the second amendment
      (refine-logs/PREREG_WWW2027_A2.md): rules Q1, Q2, Q3; M1', W1, M2'.

Stratum rows carry n_queries as the number of test triples in the stratum; each
triple is scored as a head and a tail query.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone

FB = ["fb-v1", "fb-v2", "fb-v3", "fb-v4"]
FLOCK_FB = ["fb-v1", "fb-v2"]
ULTRA_CKPTS = ["ultra_3g", "ultra_4g", "ultra_50g"]
N_DRAWS = 20
R1_MIN_POSITIVE = 15
P1_MIN_CKPTS = 2
P1_MIN_GRAPHS = 3
MIN_STRATUM_TRIPLES = 30
DETERMINISM_TOL = 1e-6
STRATA = ("inM", "notM", "inM_iso", "inM_pop")
ARMS = ("L0", "Wm", "Xm")


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def usable(r):
    return r.get("mode") == "production" and r.get("arm") != "ERROR"


def cell_of(r):
    return (r.get("construction", "legacy"), r["backbone"], r["ckpt"], r["graph"])


def collect(rows):
    """Per cell: pooled MRR by (arm, draw), stratum (MRR, n) by draw and (arm, stratum),
    and the construction statistics carried by the pooled Xm rows."""
    pooled = defaultdict(dict)
    strata = defaultdict(lambda: defaultdict(dict))
    stats = defaultdict(dict)
    for r in rows:
        if not usable(r) or r["arm"] not in ARMS:
            continue
        cell = cell_of(r)
        if r.get("stratum") is None:
            pooled[cell][(r["arm"], r["draw"])] = r["mrr"]
            if r["arm"] == "Xm":
                stats[cell][r["draw"]] = {k: r.get(k) for k in ("moved_base", "x_l1_median")}
        else:
            strata[cell][r["draw"]][(r["arm"], r["stratum"])] = (r["mrr"], r["n_queries"])
    return pooled, strata, stats


def cell_complete(draws, pooled):
    """All 20 draws, each with the pooled Wm/Xm rows, a baseline, and L0/Wm/Xm on inM and notM."""
    if sorted(draws) != list(range(N_DRAWS)):
        return False
    for d, s in draws.items():
        if ("L0", d) not in pooled and ("L0", -1) not in pooled:
            return False
        if any((arm, d) not in pooled for arm in ("Wm", "Xm")):
            return False
        if any((arm, name) not in s for arm in ARMS for name in ("inM", "notM")):
            return False
    return True


def draw_record(d, s, pooled):
    rec = {"draw": d}
    for name in STRATA:
        l0 = s.get(("L0", name))
        n = l0[1] if l0 else 0
        rec[f"n_{name}"] = n
        w, x = s.get(("Wm", name)), s.get(("Xm", name))
        ok = n > 0 and l0[0] and w and x and w[0] is not None and x[0] is not None
        rec[f"b_{name}"] = l0[0] if n > 0 else None
        rec[f"abs_{name}"] = w[0] - x[0] if ok else None
        rec[f"D_{name}"] = (w[0] - x[0]) / l0[0] if ok else None
    B = pooled.get(("L0", d), pooled.get(("L0", -1)))
    pw, px = pooled.get(("Wm", d)), pooled.get(("Xm", d))
    n_m, n_u = rec["n_inM"], rec["n_notM"]
    if B and pw is not None and px is not None and n_m and n_u:
        q = n_m / (n_m + n_u)
        rec["q"] = q
        rec["alpha"] = q * rec["b_inM"] / B
        rec["D_pool"] = (pw - px) / B
        rec["abs_pool"] = pw - px
        if rec["D_inM"] is not None and rec["D_notM"] is not None:
            rec["identity_residual"] = abs(rec["alpha"] * rec["D_inM"]
                                           + (1 - rec["alpha"]) * rec["D_notM"] - rec["D_pool"])
        gaps = []
        for arm, p in (("L0", B), ("Wm", pw), ("Xm", px)):
            a, b = s.get((arm, "inM")), s.get((arm, "notM"))
            if a and b and a[0] is not None and b[0] is not None:
                gaps.append(abs(p - (a[1] * a[0] + b[1] * b[0]) / (a[1] + b[1])))
        rec["pooled_vs_strata_gap"] = max(gaps) if gaps else None
    mass_m = n_m * rec["b_inM"] if n_m else 0
    if mass_m:
        for part in ("inM_iso", "inM_pop"):
            n = rec[f"n_{part}"]
            rec[f"share_{part}"] = n * rec[f"b_{part}"] / mass_m if n else 0.0
        rec["partition_residual"] = abs(rec["n_inM_iso"] + rec["n_inM_pop"] - n_m)
    return rec


def median_or_none(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def stratum_summary(recs, name, complete):
    """judged: the cell is complete and the stratum passes the 30-triple exposure guard."""
    vals = [r[f"D_{name}"] for r in recs if r[f"D_{name}"] is not None]
    med_n = st.median(r[f"n_{name}"] for r in recs)
    return {"median_n_triples": med_n,
            "low_information": med_n < MIN_STRATUM_TRIPLES,
            "judged": complete and med_n >= MIN_STRATUM_TRIPLES,
            "n_draws_defined": len(vals),
            "median_D": st.median(vals) if vals else None,
            "positive_draws": sum(v > 0 for v in vals),
            "median_abs": median_or_none(r[f"abs_{name}"] for r in recs)}


def r1(summary_inM, complete):
    if not complete or (summary_inM["n_draws_defined"] != N_DRAWS
                        and not summary_inM["low_information"]):
        return "incomplete"
    if not summary_inM["judged"]:
        return "not_judged"
    ok = summary_inM["median_D"] > 0 and summary_inM["positive_draws"] >= R1_MIN_POSITIVE
    return "holds" if ok else "fails"


def exp1_cells(rows):
    pooled, strata, stats = collect(rows)
    out = {}
    for cell, draws in strata.items():
        complete = cell_complete(draws, pooled[cell])
        recs = [draw_record(d, s, pooled[cell]) for d, s in sorted(draws.items())]
        summ = {name: stratum_summary(recs, name, complete) for name in STRATA}
        out[cell] = {
            "n_draws": len(recs),
            "complete": complete,
            "strata": summ,
            "R1": r1(summ["inM"], complete),
            "median_moved_base": median_or_none(v.get("moved_base")
                                                for v in stats[cell].values()),
            "median_x_partner_l1": median_or_none(v.get("x_l1_median")
                                                  for v in stats[cell].values()),
            "median_alpha": median_or_none(r.get("alpha") for r in recs),
            "median_q": median_or_none(r.get("q") for r in recs),
            "median_D_pool": median_or_none(r.get("D_pool") for r in recs),
            "median_abs_pool": median_or_none(r.get("abs_pool") for r in recs),
            "median_share_inM_iso": median_or_none(r.get("share_inM_iso") for r in recs),
            "median_share_inM_pop": median_or_none(r.get("share_inM_pop") for r in recs),
            "max_identity_residual": max((r["identity_residual"] for r in recs
                                          if r.get("identity_residual") is not None), default=None),
            "max_pooled_vs_strata_gap": max((r["pooled_vs_strata_gap"] for r in recs
                                             if r.get("pooled_vs_strata_gap") is not None),
                                            default=None),
            "max_partition_residual": max((r["partition_residual"] for r in recs
                                           if "partition_residual" in r), default=None),
            "draws": recs,
        }
    return out


def e1a_required_keys():
    keys = set()
    for c in ULTRA_CKPTS:
        for g in FB:
            keys.add((c, g, "L0", None, -1))
            for d in range(N_DRAWS):
                keys |= {(c, g, arm, None, d) for arm in ("Wm", "Xm")}
                keys |= {(c, g, arm, s, d) for arm in ARMS for s in ("inM", "notM")}
    return keys


def determinism(rows, archived_rows):
    """E1a: every pooled and inM/notM MRR of the twelve legacy ULTRA FB cells against the
    archive. Rows of other pooled arms (Xfull, L0_recheck) are compared when present."""
    compared_arms = ARMS + ("Xfull", "L0_recheck")

    def key(r):
        return (r["ckpt"], r["graph"], r["arm"], r.get("stratum"), r["draw"])

    def keep(r):
        return (usable(r) and r.get("backbone") == "ultra" and r["graph"] in FB
                and r["arm"] in compared_arms and r.get("stratum") in (None, "inM", "notM"))
    arch = {key(r): r["mrr"] for r in archived_rows if keep(r)}
    ours = {key(r): r["mrr"] for r in rows
            if keep(r) and r.get("construction", "legacy") == "legacy"}
    required = e1a_required_keys()
    missing_ours = sorted(required - set(ours), key=str)
    missing_arch = sorted(set(ours) - set(arch), key=str)
    diffs = [abs(v - arch[k]) for k, v in ours.items() if k in arch]
    max_diff = max(diffs, default=None)
    return {"n_required": len(required), "n_compared": len(diffs),
            "n_required_missing": len(missing_ours),
            "n_missing_in_archive": len(missing_arch),
            "missing_examples": [list(k) for k in (missing_ours + missing_arch)[:5]],
            "max_abs_diff": max_diff,
            "pass": (not missing_ours and not missing_arch and max_diff is not None
                     and max_diff <= DETERMINISM_TOL)}


def exp1(paths, archived_paths):
    rows = [r for p in paths for r in load_rows(p)]
    cells = exp1_cells(rows)
    e1a = {}
    if not archived_paths:
        e1a["decomposition"] = None
        e1a["withheld"] = "no archived rows supplied for the determinism check"
    else:
        e1a["determinism"] = determinism(rows, [r for p in archived_paths for r in load_rows(p)])
        if not e1a["determinism"]["pass"]:
            e1a["decomposition"] = None
            e1a["withheld"] = "determinism check failed"
        else:
            e1a["decomposition"] = {
                f"{c}|{g}": {"complete": cells[k]["complete"],
                             "D_inM": cells[k]["strata"]["inM"],
                             "D_inM_iso": cells[k]["strata"]["inM_iso"],
                             "D_inM_pop": cells[k]["strata"]["inM_pop"],
                             "median_share_inM_iso": cells[k]["median_share_inM_iso"],
                             "median_share_inM_pop": cells[k]["median_share_inM_pop"]}
                for c in ULTRA_CKPTS for g in FB if (k := ("legacy", "ultra", c, g)) in cells}

    p1_graphs = {}
    for g in FB:
        res = [cells[k]["R1"] if (k := ("populated", "ultra", c, g)) in cells else "absent"
               for c in ULTRA_CKPTS]
        if all(x in ("holds", "fails") for x in res):
            p1_graphs[g] = {"R1": res, "judged": True,
                            "passes": sum(x == "holds" for x in res) >= P1_MIN_CKPTS}
        else:
            p1_graphs[g] = {"R1": res, "judged": False}
    judged = [v for v in p1_graphs.values() if v["judged"]]
    n_pass = sum(v["passes"] for v in judged)
    p1 = ("not_judged" if len(judged) < P1_MIN_GRAPHS else
          "holds" if n_pass >= P1_MIN_GRAPHS else "fails")

    flock = {g: next((v["R1"] for (con, b, c, gg), v in cells.items()
                      if con == "populated" and b == "flock" and gg == g), "absent")
             for g in FLOCK_FB}
    if all(x == "holds" for x in flock.values()):
        p2 = "holds"
    elif all(x in ("holds", "fails") for x in flock.values()):
        p2 = "fails"
    else:
        p2 = "not_judged"

    nl25 = {f"{b}|{c}": v["R1"] for (con, b, c, g), v in sorted(cells.items())
            if con == "populated" and g == "nl-25"}
    verdict = {"E1a": e1a,
               "E1b": {"P1": p1, "P1_graphs": p1_graphs, "P1_graphs_passing": n_pass,
                       "P2": p2, "P2_cells": flock, "nl25_R1": nl25}}
    summary = {"|".join(k): {f: v for f, v in cells[k].items() if f != "draws"}
               for k in sorted(cells)}
    draws = {"|".join(k): cells[k]["draws"] for k in sorted(cells)}
    return {"verdict": verdict, "cells": summary, "draws": draws}


M1_MIN_ABS = 0.02
M1_FACTOR = 3.0
M2_MIN_TIER = 10
M2_MIN_CELLS = 4
M2_SHARE = 0.75


MERGE_SUBSETS = ("pooled", "affected", "shifted", "identical",
                 "affected_high", "affected_mid", "affected_low")


def exp2_cells(rows):
    cells = defaultdict(dict)
    for r in rows:
        if r.get("kind") == "eval" and r.get("mode", "production") == "production":
            cells[(r["graph"], r["ruleset"], r["ckpt"])][r["subset"]] = r
    out = {}
    for key, subs in cells.items():
        pooled = subs["pooled"]
        mass = pooled["n_queries_original"] * pooled["mrr_original"]
        q = {}
        for name, r in subs.items():
            if r["mrr_original"] is None or r["mrr_merged"] is None:
                q[name] = {"n": r["n_queries_original"], "missing": True}
                continue
            d_abs = r["mrr_merged"] - r["mrr_original"]
            q[name] = {"n": r["n_queries_original"], "n_merged": r["n_queries_merged"],
                       "mrr_original": r["mrr_original"], "mrr_merged": r["mrr_merged"],
                       "delta_abs": d_abs, "delta_rel": d_abs / r["mrr_original"],
                       "alpha": r["n_queries_original"] * r["mrr_original"] / mass}
        out[key] = q
    return out


def exp2(paths):
    rows = [r for p in paths for r in load_rows(p)]
    cells = exp2_cells(rows)
    ckpts = ULTRA_CKPTS
    expected = [(g, "A+B", c) for g in FB for c in ckpts]
    missing = [k for k in expected
               if k not in cells or any(s not in cells[k] for s in MERGE_SUBSETS)]
    if missing:
        return {"verdict": {"M1": "incomplete", "M2": "incomplete",
                            "missing_cells": ["|".join(k) for k in missing]},
                "cells": {"|".join(k): v for k, v in sorted(cells.items())}}
    m1 = {}
    for g in FB:
        a = st.median(abs(cells[(g, "A+B", c)]["affected"]["delta_abs"]) for c in ckpts)
        p = st.median(abs(cells[(g, "A+B", c)]["pooled"]["delta_abs"]) for c in ckpts)
        m1[g] = {"median_abs_delta_affected": a, "median_abs_delta_pooled": p,
                 "holds": a >= M1_MIN_ABS and a >= M1_FACTOR * p}
    m2_cells = []
    for g in FB:
        for c in ckpts:
            hi, lo = cells[(g, "A+B", c)]["affected_high"], cells[(g, "A+B", c)]["affected_low"]
            if (not hi.get("missing") and not lo.get("missing")
                    and hi["n"] >= M2_MIN_TIER and lo["n"] >= M2_MIN_TIER):
                m2_cells.append({"graph": g, "ckpt": c, "delta_high": hi["delta_rel"],
                                 "delta_low": lo["delta_rel"],
                                 "high_worse": hi["delta_rel"] < lo["delta_rel"]})
    n_q = len(m2_cells)
    n_hw = sum(x["high_worse"] for x in m2_cells)
    wk = {}
    aff = [abs(cells[("wk-75", "yago", c)]["affected"]["delta_abs"]) for c in ckpts
           if ("wk-75", "yago", c) in cells]
    pool = [abs(cells[("wk-75", "yago", c)]["pooled"]["delta_abs"]) for c in ckpts
            if ("wk-75", "yago", c) in cells]
    if len(aff) == len(ckpts):
        a, p = st.median(aff), st.median(pool)
        wk = {"median_abs_delta_affected": a, "median_abs_delta_pooled": p,
              "M1_condition": a >= M1_MIN_ABS and a >= M1_FACTOR * p}
    else:
        wk = {"status": "incomplete"}
    verdict = {
        "M1_by_graph": m1,
        "M1": "supported" if sum(1 for v in m1.values() if v.get("holds")) >= 3 else "not_supported",
        "M2_qualifying_cells": n_q,
        "M2_high_worse_cells": n_hw,
        "M2": ("not_judged" if n_q < M2_MIN_CELLS else
               ("supported" if n_hw >= M2_SHARE * n_q else "not_supported")),
        "wk75_descriptive": wk,
    }
    summary = {f"{g}|{rs}|{c}": v for (g, rs, c), v in sorted(cells.items())}
    return {"verdict": verdict, "m2_cells": m2_cells, "cells": summary}


A2_WK = ["wk-50", "wk-100"]
A2_FB = ["ingfb-25", "ingfb-50", "ingfb-75", "ingfb-100"]
A2_NL = ["nl-0"]
A2_MERGE_FB = ["ifb-25", "ifb-50", "ifb-75", "ifb-100"]
A2_FLOCK_CUTOFF = datetime(2026, 10, 23, tzinfo=timezone.utc)
A2_FLOCK_UNIT = "flock:populated:flock_entity.pth:{graph}:d{draw}"


def flock_done_before_cutoff(attempts_paths, graph):
    """All 20 draws of the FLOCK cell succeeded, per the attempt ledger, before the cutoff."""
    done = {}
    for p in attempts_paths:
        for e in load_rows(p):
            if e.get("event") == "success":
                done[e["cell"]] = datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%S%z")
    keys = [A2_FLOCK_UNIT.format(graph=graph, draw=d) for d in range(N_DRAWS)]
    return all(k in done and done[k] < A2_FLOCK_CUTOFF for k in keys)


def graph_verdict(cells, construction, g):
    res = [cells[k]["R1"] if (k := (construction, "ultra", c, g)) in cells else "absent"
           for c in ULTRA_CKPTS]
    if all(x in ("holds", "fails") for x in res):
        return {"R1": res, "judged": True, "passes": sum(x == "holds" for x in res) >= P1_MIN_CKPTS}
    return {"R1": res, "judged": False}


def exp3(paths, attempts_paths):
    rows = [r for p in paths for r in load_rows(p)]
    cells = exp1_cells(rows)
    in_time = {g: flock_done_before_cutoff(attempts_paths, g) for g in A2_WK}
    for (con, b, c, g), v in cells.items():
        if con == "populated" and b == "flock" and g in A2_WK and not in_time[g]:
            v["complete"], v["R1"] = False, "incomplete"
            for summ in v["strata"].values():
                summ["judged"] = False
    graphs = {g: graph_verdict(cells, "populated", g) for g in A2_WK + A2_FB}
    wk = [graphs[g] for g in A2_WK]
    q1 = ("not_judged" if not all(v["judged"] for v in wk) else
          "holds" if all(v["passes"] for v in wk) else "fails")
    fb = [graphs[g] for g in A2_FB if graphs[g]["judged"]]
    q2 = ("not_judged" if len(fb) < P1_MIN_GRAPHS else
          "holds" if sum(v["passes"] for v in fb) >= P1_MIN_GRAPHS else "fails")
    flock = {g: next((v["R1"] for (con, b, c, gg), v in cells.items()
                      if con == "populated" and b == "flock" and gg == g), "absent")
             for g in A2_WK}
    if all(x == "holds" for x in flock.values()):
        q3 = "holds"
    elif all(x in ("holds", "fails") for x in flock.values()):
        q3 = "fails"
    else:
        q3 = "not_judged"
    nl0 = {f"{b}|{c}": v["R1"] for (con, b, c, g), v in sorted(cells.items())
           if con == "populated" and g in A2_NL}
    verdict = {"Q1": q1, "Q2": q2, "Q2_graphs_passing": sum(v["passes"] for v in fb),
               "Q3": q3, "Q3_cells": flock, "Q3_complete_before_cutoff": in_time,
               "graphs": graphs, "nl0_R1": nl0}
    summary = {"|".join(k): {f: v for f, v in cells[k].items() if f != "draws"}
               for k in sorted(cells)}
    draws = {"|".join(k): cells[k]["draws"] for k in sorted(cells)}
    return {"verdict": verdict, "cells": summary, "draws": draws}


def local_condition(cells, g, rs):
    """M1's condition on one graph; None when the graph contributes no complete cell."""
    cs = [cells.get((g, rs, c)) for c in ULTRA_CKPTS]
    if any(c is None or any(s not in c for s in MERGE_SUBSETS) for c in cs):
        return None
    if any(c["affected"].get("missing") for c in cs):
        return None
    a = st.median(abs(c["affected"]["delta_abs"]) for c in cs)
    p = st.median(abs(c["pooled"]["delta_abs"]) for c in cs)
    return {"median_abs_delta_affected": a, "median_abs_delta_pooled": p,
            "holds": a >= M1_MIN_ABS and a >= M1_FACTOR * p}


def exp4(paths):
    rows = [r for p in paths for r in load_rows(p)]
    cells = exp2_cells(rows)
    fb_ab = [(g, "A+B", c) for g in A2_MERGE_FB for c in ULTRA_CKPTS]
    wk = [(g, "yago", c) for g in A2_WK for c in ULTRA_CKPTS]
    missing = sorted(k for k in fb_ab + wk + [(g, "A", c) for g in A2_MERGE_FB for c in ULTRA_CKPTS]
                     if k not in cells or any(s not in cells[k] for s in MERGE_SUBSETS))

    def complete(keys):
        return not any(k in missing for k in keys)
    m1 = {g: local_condition(cells, g, "A+B") for g in A2_MERGE_FB} if complete(fb_ab) else {}
    w1 = {g: local_condition(cells, g, "yago") for g in A2_WK} if complete(wk) else {}
    m1p = ("incomplete" if not complete(fb_ab) else
           "supported" if sum(1 for v in m1.values() if v and v["holds"]) >= 3 else "not_supported")
    w1v = ("incomplete" if not complete(wk) else
           "supported" if all(v and v["holds"] for v in w1.values()) else "not_supported")
    m2_cells = []
    for g, rs in [(g, "A+B") for g in A2_MERGE_FB] + [(g, "yago") for g in A2_WK]:
        for c in ULTRA_CKPTS:
            sub = cells.get((g, rs, c))
            if not sub or any(x not in sub for x in MERGE_SUBSETS):
                continue
            hi, lo = sub["affected_high"], sub["affected_low"]
            if (not hi.get("missing") and not lo.get("missing")
                    and hi["n"] >= M2_MIN_TIER and lo["n"] >= M2_MIN_TIER):
                m2_cells.append({"graph": g, "ckpt": c, "delta_high": hi["delta_rel"],
                                 "delta_low": lo["delta_rel"],
                                 "high_worse": hi["delta_rel"] < lo["delta_rel"]})
    n_q, n_hw = len(m2_cells), sum(x["high_worse"] for x in m2_cells)
    m2p = ("incomplete" if not complete(fb_ab + wk) else
           "not_judged" if n_q < M2_MIN_CELLS else
           "supported" if n_hw >= M2_SHARE * n_q else "not_supported")
    verdict = {"M1p": m1p, "M1p_by_graph": m1, "W1": w1v, "W1_by_graph": w1, "M2p": m2p,
               "M2p_qualifying_cells": n_q, "M2p_high_worse_cells": n_hw,
               "missing_cells": ["|".join(k) for k in missing],
               "graphs_without_affected_targets": sorted(
                   {g for (g, rs, c) in fb_ab + wk
                    if cells.get((g, rs, c), {}).get("affected", {}).get("missing")})}
    summary = {"|".join(k): v for k, v in sorted(cells.items())}
    return {"verdict": verdict, "m2_cells": m2_cells, "cells": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["exp1", "exp2", "exp3", "exp4"])
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--archived", nargs="*", default=[],
                    help="exp1: archived matched-contrast rows for the E1a determinism check")
    ap.add_argument("--attempts", nargs="*", default=[],
                    help="exp3: the FLOCK run's attempt ledger, for the completion cutoff")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = {"exp1": lambda: exp1(args.jsonl, args.archived), "exp2": lambda: exp2(args.jsonl),
           "exp3": lambda: exp3(args.jsonl, args.attempts), "exp4": lambda: exp4(args.jsonl)}
    result = run[args.experiment]()
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
