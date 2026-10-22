"""matched_feasibility.py — EXPLORATORY INSTRUMENT-DESIGN STATISTICS (PLAN_AMENDMENT_3 prep, R2).

Computes, per claim graph and backbone, the displacement-matched pair-pool sizes and
exposure statistics for the proposed Wm/Xm matched contrast (within-role vs cross-role
transpositions at IDENTICAL moved set, fixed points, cycle type, and atomic base/inverse
lift), over the EXACT production draw-seed sets (design review R1 Q3/Q8 fix):

  units          : disjoint same-class base-relation pairs (the Wm arm's transpositions)
  |M|            : moved BASE relations after X-matchability trim (moved nodes = 2|M|)
  f_matched      : |M| / R_base (= moved-node fraction of the 2R relation-graph nodes)
  rg_incidence   : fraction of relation-graph edges with >=1 endpoint in lifted moved set
  query_mass     : fraction of test target triples whose relation id is in the lifted set
  msg_mass       : (FLOCK) fraction of message edges whose edge_type is in the lifted set
  f_Xfull        : moved fraction of the Xfull descriptive dose anchor (all-base cross pairing)
  judgeable      : min-over-seeds f_matched >= 0.10  AND  not LOW_EXPOSURE
  LOW_EXPOSURE   : MEDIAN-over-scheduled-draws of the CHANNEL-PROXIMAL exposure < 0.10
                   (ULTRA/TRIX channel = rg_incidence; FLOCK channel = msg_mass);
                   query_mass < 0.05 is a REPORT-ONLY flag, never eligibility (R3 rule)

Matching algorithm (frozen, named honestly per design review R1 Q8): best-of-20 seeded randomized
greedy MAXIMAL matching — NOT guaranteed maximum; deterministic given (inputs, seed,
numpy PCG64). Underfill shrinks the shared pool only; it cannot favor either arm.

Certification (design review R1 Q1 fix): the LIFTED permutations are constructed and certified in
full 2R space — permutation validity, involution pi[pi]==id, exact moved set == M ∪ (M+R),
exactly |M| lifted 2-cycles per arm... (2 per base pair), and atomicity pi[r+R]==pi[r]+R.

DISCIPLINE: loads claim graphs with allow_claim=True for STRUCTURAL STATISTICS ONLY —
zero checkpoint/model contact, no MRR computed anywhere in this file. Disclosed in
PLAN_AMENDMENT_3 §I.

Run (CPU only):
  CUDA_VISIBLE_DEVICES="" python matched_feasibility.py [--backbones ultra,trix,flock]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse, copy, json
import numpy as np
import torch

# production draw-seed sets (PLAN_AMENDMENT_3 §D; design review R1 Q3 fix: judgeability is
# defined over the EXACT scheduled sets, not a 3-seed sample)
PROD_SEEDS = {"ultra": list(range(20)), "trix": list(range(20)),
              "flock": list(range(10))}


# ---------------------------------------------------------------- construction ----

def _classes_of(prof, R_base, joint):
    """Group BASE relation ids by joint (base,inverse) profile key (joint=True)
    or by base-side profile only (joint=False; FLOCK's frozen base-space Pi)."""
    classes = {}
    for r in range(R_base):
        if joint:
            key = (tuple(prof[r].tolist()), tuple(prof[r + R_base].tolist()))
        else:
            key = tuple(prof[r].tolist())
        classes.setdefault(key, []).append(r)
    return classes


def _w_units(classes, rng):
    """Seeded within-class disjoint pairs (one odd member per class dropped)."""
    units = []
    for members in classes.values():
        if len(members) < 2:
            continue
        mem = list(rng.permutation(members))
        for i in range(0, len(mem) - 1, 2):
            units.append((int(mem[i]), int(mem[i + 1])))
    return units


def _cross_ok(prof, R_base, x, y, strong):
    """Element-level cross condition for an Xm transposition (x,y):
    strong (ULTRA/TRIX): base-side AND inverse-side marginal profiles differ;
    else (FLOCK Pi semantics): base-side profile differs."""
    if tuple(prof[x].tolist()) == tuple(prof[y].tolist()):
        return False
    if strong and tuple(prof[x + R_base].tolist()) == tuple(prof[y + R_base].tolist()):
        return False
    return True


def _unit_pair_options(prof, R_base, u, v, strong):
    """Allowed element pairings when X-matching W-units u=(a,b), v=(c,d)."""
    a, b = u
    c, d = v
    opts = []
    if _cross_ok(prof, R_base, a, c, strong) and _cross_ok(prof, R_base, b, d, strong):
        opts.append(((a, c), (b, d)))
    if _cross_ok(prof, R_base, a, d, strong) and _cross_ok(prof, R_base, b, c, strong):
        opts.append(((a, d), (b, c)))
    return opts


def _greedy_maximal_matching(n_nodes, adj, rng, restarts=20):
    """Best-of-`restarts` seeded randomized greedy MAXIMAL matching (frozen; not
    guaranteed maximum). Deterministic given rng state sequence."""
    best = []
    order0 = np.arange(n_nodes)
    for _ in range(restarts):
        order = rng.permutation(order0)
        used = np.zeros(n_nodes, dtype=bool)
        m = []
        for i in order:
            if used[i]:
                continue
            nbrs = [j for j in adj[i] if not used[j]]
            if not nbrs:
                continue
            j = nbrs[int(rng.integers(len(nbrs)))]
            used[i] = used[j] = True
            m.append((int(i), int(j)))
        if len(m) > len(best):
            best = m
    return best


def _lift(pairs, R_base):
    """Full 2R permutation from base transposition list, atomic lift."""
    n = 2 * R_base
    pi = np.arange(n)
    for a, b in pairs:
        pi[a], pi[b] = b, a
        pi[a + R_base], pi[b + R_base] = b + R_base, a + R_base
    return pi


def _certify_lifted(pi, pairs, R_base, tag):
    """Full-space certificate (design review R1 Q1): permutation validity, involution,
    exact moved set == M ∪ (M+R), 2|pairs| lifted 2-cycles, atomicity."""
    n = 2 * R_base
    assert len(np.unique(pi)) == n, f"{tag}: not a permutation"
    assert np.array_equal(pi[pi], np.arange(n)), f"{tag}: not an involution"
    M = sorted({r for p in pairs for r in p})
    moved = set(np.nonzero(pi != np.arange(n))[0].tolist())
    want = set(M) | {r + R_base for r in M}
    assert moved == want, f"{tag}: moved set != M ∪ (M+R)"
    n_two_cycles = int((pi > np.arange(n))[list(moved)].sum()) if moved else 0
    assert n_two_cycles == 2 * len(pairs), f"{tag}: lifted 2-cycle count mismatch"
    base = np.arange(R_base)
    assert np.array_equal(pi[base + R_base], pi[base] + R_base), f"{tag}: lift not atomic"
    return M


def _profile_l1(prof, x, y, R_base):
    b = int(torch.abs(prof[x] - prof[y]).sum())
    i = int(torch.abs(prof[x + R_base] - prof[y + R_base]).sum())
    return b + i


def build_matched_pair(prof, R_base, seed, joint, strong, compute_xfull=True):
    """One draw: construct + certify (pi_W, pi_X) and return pool/exposure raw sets.
    compute_xfull=False for FLOCK (Xfull is an ULTRA/TRIX-only descriptive arm)."""
    rng = np.random.default_rng(seed)
    classes = _classes_of(prof, R_base, joint=joint)
    units = _w_units(classes, rng)
    nu = len(units)
    cls_of = {}
    for key, members in classes.items():
        for r in members:
            cls_of[r] = key
    adj = [[] for _ in range(nu)]
    for i in range(nu):
        for j in range(i + 1, nu):
            if cls_of[units[i][0]] == cls_of[units[j][0]]:
                continue
            if _unit_pair_options(prof, R_base, units[i], units[j], strong):
                adj[i].append(j)
                adj[j].append(i)
    matching = _greedy_maximal_matching(nu, adj, rng)
    kept_units = sorted({i for ij in matching for i in ij})
    x_pairs = []
    for i, j in matching:
        opts = _unit_pair_options(prof, R_base, units[i], units[j], strong)
        x_pairs.extend(opts[int(rng.integers(len(opts)))])
    w_pairs = [units[i] for i in kept_units]
    pi_w = _lift(w_pairs, R_base)
    pi_x = _lift(x_pairs, R_base)
    M_w = _certify_lifted(pi_w, w_pairs, R_base, "Wm")
    M_x = _certify_lifted(pi_x, x_pairs, R_base, "Xm")
    assert M_w == M_x, "moved-set mismatch between arms"
    for a, b in w_pairs:
        assert cls_of[a] == cls_of[b], "W pair crosses class"
        # joint profile distance is 0 for ULTRA/TRIX by the joint key; for FLOCK
        # (base-only classes) it holds via the verified partition equality — assert
        # directly in both cases as an empirical-invariant guard (design review R2 N4).
        assert _profile_l1(prof, a, b, R_base) == 0, "W pair joint profile dist != 0"
    for a, b in x_pairs:
        assert _cross_ok(prof, R_base, a, b, strong), "X pair not cross"
    x_dist = [_profile_l1(prof, a, b, R_base) for a, b in x_pairs]
    # Xfull descriptive dose anchor (ULTRA/TRIX only): cross pairing over ALL base
    # relations — lifted + certified like the claim arms (design review R2 N4 / R3 S3)
    xfull_pairs = []
    pi_xfull = None
    if compute_xfull:
        all_adj = [[] for _ in range(R_base)]
        for x in range(R_base):
            for y in range(x + 1, R_base):
                if _cross_ok(prof, R_base, x, y, strong):
                    all_adj[x].append(y)
                    all_adj[y].append(x)
        xfull_pairs = _greedy_maximal_matching(R_base, all_adj, rng)
        pi_xfull = _lift(xfull_pairs, R_base)
        _certify_lifted(pi_xfull, xfull_pairs, R_base, "Xfull")
        for a, b in xfull_pairs:
            assert _cross_ok(prof, R_base, a, b, strong), "Xfull pair not cross"
    return {
        "n_units_raw": nu,
        "n_units_kept": len(kept_units),
        "moved_base": len(M_w),
        "f_matched": len(M_w) / max(R_base, 1),
        "M": M_w,
        "x_pair_profile_l1_median": float(np.median(x_dist)) if x_dist else 0.0,
        "x_pair_profile_l1_q25": float(np.percentile(x_dist, 25)) if x_dist else 0.0,
        "x_pair_profile_l1_q75": float(np.percentile(x_dist, 75)) if x_dist else 0.0,
        "f_Xfull": 2 * len(xfull_pairs) / max(R_base, 1),
        "n_classes": len(classes),
        "permutable_base": sum(len(m) for m in classes.values() if len(m) > 1),
        "largest_class": max((len(m) for m in classes.values()), default=0),
        # production-consumer fields (matched_pairs.make_arms); popped before
        # JSON aggregation in _per_graph
        "pi_w": pi_w, "pi_x": pi_x, "pi_xfull": pi_xfull,
        "w_pairs": w_pairs, "x_pairs": x_pairs, "x_dist": x_dist,
    }


# ------------------------------------------------------------------ exposure ----

def _exposure(rg_ei, target_et, M, R_base, msg_et=None):
    """rg_incidence: fraction of relation-graph edges with >=1 endpoint in the lifted
    moved set; query_mass: fraction of test target triples whose relation id is in the
    lifted set; msg_mass (FLOCK): fraction of message edges with edge_type in the set."""
    lifted = torch.zeros(2 * R_base, dtype=torch.bool)
    if M:
        idx = torch.tensor(M, dtype=torch.long)
        lifted[idx] = True
        lifted[idx + R_base] = True
    if rg_ei is not None and rg_ei.numel():
        inc = (lifted[rg_ei[0]] | lifted[rg_ei[1]]).float().mean().item()
    else:
        inc = 0.0
    qm = lifted[target_et].float().mean().item() if target_et.numel() else 0.0
    out = {"rg_incidence": inc, "query_mass": qm}
    if msg_et is not None:
        out["msg_mass"] = lifted[msg_et].float().mean().item() if msg_et.numel() else 0.0
    return out


# ------------------------------------------------------------------- backbones ----

ULTRA_GRAPHS = [("fb", "v1"), ("fb", "v2"), ("fb", "v3"), ("fb", "v4"),
                ("nl", "50"), ("nl", "75"), ("nl", "100"), ("wk", "75")]
TRIX_GRAPHS = ULTRA_GRAPHS
FLOCK_GRAPHS = [("nl", "50"), ("nl", "75"), ("nl", "100"), ("wk", "75")]

# Eligibility (design review R2 N1 fix): CHANNEL-PROXIMAL exposure, aggregated by the
# MEDIAN over the scheduled draws (aligned with the median estimand C-10) —
# ULTRA/TRIX consume the (4-type) relation graph -> channel = rg_incidence;
# FLOCK consumes raw message edge_type -> channel = msg_mass.
# query_mass is a REPORT-ONLY stratum-size flag (context effects propagate to
# unmoved-relation queries through the consumed channel), never eligibility.
F_MIN, CH_MED_MIN, QMASS_FLAG = 0.10, 0.10, 0.05


def _partition_equal(prof, R_base):
    """True iff the joint (base,inverse) partition equals the base-only partition
    (design review R1 Q2: verify the claimed no-op instead of asserting it)."""
    a = _classes_of(prof, R_base, joint=True)
    b = _classes_of(prof, R_base, joint=False)
    sa = {frozenset(v) for v in a.values()}
    sb = {frozenset(v) for v in b.values()}
    return sa == sb


def _per_graph(prof, R_base, seeds, joint, strong, rg_ei, target_et,
               channel="rg_incidence", msg_et=None, compute_xfull=True):
    per = []
    for s in seeds:
        r = build_matched_pair(prof, R_base, s, joint=joint, strong=strong,
                               compute_xfull=compute_xfull)
        for heavy in ("pi_w", "pi_x", "pi_xfull", "w_pairs", "x_pairs", "x_dist"):
            r.pop(heavy, None)
        r.update(_exposure(rg_ei, target_et, r.pop("M"), R_base, msg_et=msg_et))
        per.append(r)
    keys = ["moved_base", "f_matched", "n_units_kept", "rg_incidence",
            "query_mass", "x_pair_profile_l1_median",
            "x_pair_profile_l1_q25", "x_pair_profile_l1_q75"]
    if compute_xfull:
        keys.append("f_Xfull")
    if msg_et is not None:
        keys.append("msg_mass")
    rep = {"R_base": R_base, "seeds": list(seeds), "channel": channel,
           "n_classes": per[0]["n_classes"],
           "permutable_base": per[0]["permutable_base"],
           "largest_class": per[0]["largest_class"],
           "joint_equals_base_partition": _partition_equal(prof, R_base)}
    for k in keys:
        vals = [p[k] for p in per]
        rep[k + "_min"] = min(vals)
        rep[k + "_max"] = max(vals)
        rep[k + "_median"] = float(np.median(vals))
        rep[k + "_per_seed"] = vals
    rep["low_exposure"] = rep[channel + "_median"] < CH_MED_MIN
    rep["low_query_mass_flag"] = rep["query_mass_median"] < QMASS_FLAG  # report-only
    rep["judgeable"] = (rep["f_matched_min"] >= F_MIN) and not rep["low_exposure"]
    return rep


def stats_ultra():
    import pilot_harness as PH
    import tunit_common as TC
    from ultra.tasks import build_relation_graph
    out = {}
    for name, ver in ULTRA_GRAPHS:
        if name == "fb":
            data, _f = PH.load_inductive("fb", ver)
        else:
            # structural-statistics-only claim-graph access; zero model contact
            data, _f = TC.load_ingram(name, ver, allow_claim=True)
        d = copy.copy(data).cpu()
        if hasattr(d, "relation_graph"):
            del d.relation_graph
        d = build_relation_graph(d)
        rg = d.relation_graph
        prof = TC.rg_degree_profile(rg)
        n = prof.shape[0]
        assert n % 2 == 0, (name, ver, n)
        out[f"{name}-{ver}"] = _per_graph(
            prof, n // 2, PROD_SEEDS["ultra"], joint=True, strong=True,
            rg_ei=rg.edge_index.cpu(), target_et=d.target_edge_type.cpu())
    return out


def stats_trix():
    import trix_harness as TH
    TH.DEVICE = torch.device("cpu")
    out = {}
    for name, ver in TRIX_GRAPHS:
        if name == "fb":
            data, _f = TH.load_inductive_fb(ver)
        else:
            data, _f = TH.load_inductive_ingram(name, ver)
        prof = TH.relgraph_degree_profile(data.relation_adj)
        n = prof.shape[0]
        assert n % 2 == 0, (name, ver, n)
        rg_ei = torch.cat([data.relation_adj[k].edge_index.cpu()
                           for k in ("hh", "ht", "th", "tt")
                           if data.relation_adj[k].edge_index.numel()], dim=1)
        out[f"{name}-{ver}"] = _per_graph(
            prof, n // 2, PROD_SEEDS["trix"], joint=True, strong=True,
            rg_ei=rg_ei, target_et=data.target_edge_type.cpu())
    return out


def stats_flock():
    import flock_ladder as FL
    FL.DEVICE = torch.device("cpu")
    out = {}
    for name, ver in FLOCK_GRAPHS:
        # structural-statistics-only claim-graph access; zero model contact
        cfg, test, _filt = FL.build_graph(name, ver, allow_claim=True)
        R_base = int(test.num_relations) // 2
        num_rels = int(test.num_relations)
        rg_ei, rg_et = FL.build_relation_graph_edges(
            test.edge_index, test.edge_type, int(test.num_nodes), num_rels)
        prof = FL.rg_degree_profile(rg_ei, rg_et, num_rels)
        # FLOCK frozen Pi = base-side classes; X = base-side cross (strong=False)
        out[f"{name}-{ver}"] = _per_graph(
            prof, R_base, PROD_SEEDS["flock"], joint=False, strong=False,
            rg_ei=rg_ei.cpu(), target_et=test.target_edge_type.cpu(),
            channel="msg_mass", msg_et=test.edge_type.cpu(), compute_xfull=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="ultra,trix,flock")
    ap.add_argument("--out", default="runs/MATCHED_FEASIBILITY/feasibility_r3.json")
    args = ap.parse_args()
    todo = args.backbones.split(",")
    result = {"note": "Structural stats only; no "
                      "checkpoint/model contact; judgeability over EXACT production "
                      "seed sets " + json.dumps(PROD_SEEDS),
              "rule": "low_exposure := median(channel exposure over scheduled draws)"
                      " < 0.10 with channel rg_incidence (ULTRA/TRIX) / msg_mass "
                      "(FLOCK); judgeable := f_matched_min >= 0.10 and not "
                      "low_exposure; query_mass_median < 0.05 is report-only flag",
              "thresholds": {"f_min": F_MIN, "channel_median_min": CH_MED_MIN,
                             "query_mass_flag": QMASS_FLAG},
              "numpy_version": np.__version__, "torch_version": torch.__version__}
    if "ultra" in todo:
        result["ultra"] = stats_ultra()
    if "trix" in todo:
        result["trix"] = stats_trix()
    if "flock" in todo:
        result["flock"] = stats_flock()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    for bk in ("ultra", "trix", "flock"):
        if bk not in result:
            continue
        print(f"== {bk}")
        for g, r in result[bk].items():
            print(f"  {g:8s} R={r['R_base']:4d} "
                  f"|M|=[{r['moved_base_min']},{r['moved_base_max']}] "
                  f"f=[{r['f_matched_min']:.3f},{r['f_matched_max']:.3f}] "
                  f"ch({r['channel']})med={r[r['channel'] + '_median']:.3f} "
                  f"qMassMed={r['query_mass_median']:.3f}"
                  f"{'(FLAG)' if r['low_query_mass_flag'] else ''} "
                  + (f"fXfull>={r['f_Xfull_min']:.2f} " if 'f_Xfull_min' in r else "")
                  + f"jointEqBase={r['joint_equals_base_partition']} "
                  f"lowExp={r['low_exposure']} judgeable={r['judgeable']}")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
