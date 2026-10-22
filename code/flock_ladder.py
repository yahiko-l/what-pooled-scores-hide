"""FLOCK matched-control ladder — PRODUCTION runner (pre-freeze build + NL-25 validation).

Ports the ULTRA/TRIX matched-control instrument to the FLOCK backbone. Unlike
ULTRA/TRIX (whose entity model reads a *separate* relation graph, so the control
relabels that graph's nodes), FLOCK has NO model-internal relation graph: it walks
the entity graph and binds the query relation to structure via a raw-type identity
marker (`is_r`, models.py:534-540) + a per-relation state readout. The smoke
(flock_harness.py / flock_smoke_nl25.json) validated that permuting the message-graph
`edge_type` (query + filter fixed) is the faithful FLOCK analog of the ULTRA/TRIX
relation-node relabel. This runner completes the matched PAIR by adding the
Pi-based role-PRESERVING arm.

Arms (per graph, CRN-paired to a base walk seed S):
  L0        base (no relabel) — 3 seeds give the L0-L0 invariance jitter band.
  L4loose   ROLE-BREAK: sigma = arbitrary permutation of BASE relation ids
            (= smoke's D-ii). Applied to message edge_type only; query+filter fixed.
  L4        ROLE-PRESERVING: sigma shuffles BASE relations only WITHIN their
            degree-profile role class Pi (singletons fixed). Applied to message
            edge_type only; query+filter fixed.
  D-i       EQUIVARIANCE CONTROL: the SAME arbitrary pi as L4loose applied
            consistently to message + query + filter. Expect recovery ~ 1.

Pi (role partition) is computed EXACTLY as ULTRA/TRIX/tunit_common:
  build the 4-type relation graph from the message graph (relations = nodes;
  hh/tt = share a head/tail entity, ht/th = cross), take the 8-dim typed
  degree profile (4 types x out/in), group BASE relations with equal profile.

Inverse convention (from the loaded Data, verified in the smoke): message
edge_type is base+inverse over {0..2R-1}, R=num_relations//2; queries are
base-only; inverse(r)=r+R (models.py:82/642). Every base permutation sigma is
lifted to pi over base+inverse by pi(r)=sigma(r) (base), sigma(r-R)+R (inverse),
preserving base/inverse pairing.

CRN: before EVERY eval, random/numpy/torch/torch.cuda are reset to the SAME seed S
(reuses flock_harness.set_all_seeds). Eval MRR is FLOCK's OWN ranking (run.test),
reused verbatim. Decision rules are MECHANICAL and written to the summary.

INTEGRITY: nothing under FLOCK/src or the .pth checkpoints is modified; this file
only READS them. Claim (OOF) graphs {nl-50,nl-75,nl-100,wk-75} are frozen and
guarded; this pre-freeze build validates ONLY on the exploration graph NL-25.
"""
import os
import sys
import json
import time
import copy
import argparse

import numpy as np
import torch
from torch_geometric.data import Data

import flock_harness as FH  # reuses build/eval/CRN/perm helpers; does NOT run its main()

_PILOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- frozen production-claim invocation (mirrors trix_ladder.TRIX_CLAIM_FROZEN) ----
CLAIM_GRAPH_NAMES = {"nl-50", "nl-75", "nl-100", "wk-75"}
EXPLORE_GRAPH_NAMES = {"nl-25", "wk-25", "nl-0"}
ARMS_ALL = ["L4loose", "L4", "D-i"]
FLOCK_CLAIM_FROZEN = {
    "graphs": {"nl-50", "nl-75", "nl-100", "wk-75"},
    "arms": {"L4", "L4loose", "D-i"},
    "seeds": 5,
    "ckpt": "flock_entity.pth",
    "crn": "reset random/numpy/torch/torch.cuda to seed S before EACH eval (paired)",
    "config": "src_entity/config/zeroshot_inductive/n16_ensemble16.yaml",
    "eval": "FLOCK run.test filtered ranking, verbatim",
}

# mechanical decision constants
COLLAPSE_MARGIN = 0.10   # L4loose 'collapses' if median recovery < band_lo - this
BAND_TOL = 0.03          # walk-noise tolerance around the L0-L0 recovery band
DISSOC_MARGIN = 0.30     # L4 'preserved' also accepted if L4 - L4loose >= this


def validate_flock_claim_invocation(args, arms):
    """Enforce the frozen invocation whenever a claim (OOF) graph is requested.
    Returns 'production' | 'smoke' | None. Mirrors
    trix_ladder.validate_trix_claim_invocation."""
    req = {g for g in args.claim_graphs.split(",") if g}
    if not req:
        return None
    if not (req & CLAIM_GRAPH_NAMES):
        # exploration-graph exercise of the claim code path — never evidence
        return "smoke"
    problems = []
    if req != FLOCK_CLAIM_FROZEN["graphs"]:
        problems.append(f"graphs {sorted(req)} != frozen "
                        f"{sorted(FLOCK_CLAIM_FROZEN['graphs'])} (all four, one shot)")
    if set(arms) != FLOCK_CLAIM_FROZEN["arms"]:
        problems.append(f"arms {sorted(set(arms))} != frozen "
                        f"{sorted(FLOCK_CLAIM_FROZEN['arms'])}")
    if args.seeds != FLOCK_CLAIM_FROZEN["seeds"]:
        problems.append(f"seeds {args.seeds} != {FLOCK_CLAIM_FROZEN['seeds']}")
    if os.path.basename(args.ckpt) != FLOCK_CLAIM_FROZEN["ckpt"]:
        problems.append(f"ckpt {os.path.basename(args.ckpt)} != frozen "
                        f"{FLOCK_CLAIM_FROZEN['ckpt']}")
    if args.limit:
        problems.append("--limit forbidden in production claim mode")
    if problems:
        raise SystemExit("claim-mode invocation violates FLOCK freeze "
                         "(FLOCK_CLAIM_FROZEN): " + "; ".join(problems))
    return "production"


# ----------------------------------------------------------- Pi role partition ----

def build_relation_graph_edges(edge_index, edge_type, num_nodes, num_rels):
    """ULTRA-convention 4-type relation graph (unique-pair), computed from the
    message graph. Returns (rg_edge_index[2,K], rg_edge_type[K]) on CPU with
    edge_type in {0:hh, 1:tt, 2:ht, 3:th} — matching ultra.tasks.build_relation_graph
    and tunit_common.rg_degree_profile's type order."""
    ei = edge_index.cpu()
    et = edge_type.cpu()
    Eh = torch.stack([ei[0], et]).t().unique(dim=0)   # unique (head, rel)
    Et = torch.stack([ei[1], et]).t().unique(dim=0)   # unique (tail, rel)

    def incidence(E):  # node x rel, 1 where relation incident on node
        vals = torch.ones(E.shape[0])
        return torch.sparse_coo_tensor(E.t(), vals, (num_nodes, num_rels)).coalesce()

    Eh_sp, Et_sp = incidence(Eh), incidence(Et)
    EhT, EtT = Eh_sp.t(), Et_sp.t()                    # rel x node
    Ahh = torch.sparse.mm(EhT, Eh_sp).coalesce()       # share a head entity
    Att = torch.sparse.mm(EtT, Et_sp).coalesce()       # share a tail entity
    Aht = torch.sparse.mm(EhT, Et_sp).coalesce()       # r1 head, r2 tail
    Ath = torch.sparse.mm(EtT, Eh_sp).coalesce()       # r1 tail, r2 head

    parts_i, parts_t = [], []
    for A, t in [(Ahh, 0), (Att, 1), (Aht, 2), (Ath, 3)]:
        idx = A.indices()
        parts_i.append(idx)
        parts_t.append(torch.full((idx.shape[1],), t, dtype=torch.long))
    return torch.cat(parts_i, dim=1), torch.cat(parts_t)


def rg_degree_profile(rg_ei, rg_et, n):
    """(n, 8) typed degree profile: cols (out_hh,out_tt,out_ht,out_th, in_hh,in_tt,
    in_ht,in_th). Mirrors tunit_common.rg_degree_profile exactly."""
    prof = torch.zeros(n, 8, dtype=torch.long)
    for t in range(4):
        m = rg_et == t
        k = int(m.sum())
        if k == 0:
            continue
        prof[:, t].scatter_add_(0, rg_ei[0][m], torch.ones(k, dtype=torch.long))
        prof[:, 4 + t].scatter_add_(0, rg_ei[1][m], torch.ones(k, dtype=torch.long))
    return prof


def role_partition_base(test_data, R_base):
    """Compute Pi role classes over BASE relations {0..R_base-1} from the message
    graph's 4-type relation graph. Returns (classes dict, profile tensor, stats)."""
    num_rels = int(test_data.num_relations)
    num_nodes = int(test_data.num_nodes)
    rg_ei, rg_et = build_relation_graph_edges(
        test_data.edge_index, test_data.edge_type, num_nodes, num_rels)
    prof = rg_degree_profile(rg_ei, rg_et, num_rels)
    classes = {}
    for r in range(R_base):
        classes.setdefault(tuple(prof[r].tolist()), []).append(r)
    n_permutable = sum(len(m) for m in classes.values() if len(m) > 1)
    stats = {
        "n_role_classes": len(classes),
        "n_base_relations": R_base,
        "permutable_base_relations": int(n_permutable),
        "permutable_fraction": float(n_permutable) / max(R_base, 1),
        "n_singleton_classes": sum(1 for m in classes.values() if len(m) == 1),
        "largest_class": max((len(m) for m in classes.values()), default=0),
        "relation_graph_edges": int(rg_ei.shape[1]),
    }
    return classes, prof, stats


def _lift_to_pi(sigma_base, R_base):
    """Lift a BASE permutation sigma (len R_base) to pi over base+inverse (len 2R),
    pairing-preserving: pi(r)=sigma(r) for base, sigma(r-R)+R for inverse."""
    sigma = np.asarray(sigma_base)
    pi = np.concatenate([sigma, sigma + R_base])
    assert len(np.unique(pi)) == 2 * R_base, "pi is not a permutation"
    return pi


def make_role_perm(test_data, R_base, seed):
    """L4 role-PRESERVING permutation: shuffle base relations only within their
    Pi degree-profile class (singletons fixed). Returns (pi[np, len 2R], stats)."""
    classes, _prof, stats = role_partition_base(test_data, R_base)
    rng = np.random.default_rng(seed)
    sigma = np.arange(R_base)
    for members in classes.values():
        if len(members) > 1:
            shuffled = rng.permutation(members)
            for a, b in zip(members, shuffled):
                sigma[a] = b
    assert len(np.unique(sigma)) == R_base
    pi = _lift_to_pi(sigma, R_base)
    stats = dict(stats, arm="L4", seed=int(seed),
                 sigma_moves=int((sigma != np.arange(R_base)).sum()))
    return pi, stats


def make_loose_perm(R_base, seed):
    """L4loose role-BREAK permutation: arbitrary base permutation (= smoke D-ii).
    Reuses flock_harness.make_perms so nl-25 L4loose reproduces the smoke exactly."""
    _sigma, pi = FH.make_perms(R_base, torch.device("cpu"), seed)
    pi_np = pi.cpu().numpy()
    stats = {"arm": "L4loose", "seed": int(seed), "permutable_fraction": 1.0,
             "sigma_moves": int((pi_np[:R_base] != np.arange(R_base)).sum())}
    return pi_np, stats


def query_relation_coverage_after_perm(test_data, R_base, arm, seed):
    """Report-only power/coverage flag (review MINOR / §D). Fraction of the
    unique test query relations that STILL have >=1 supporting edge in the
    message graph AFTER the arm's relabel pi is applied. For L4/L4loose the query
    ids stay original (message-only relabel); for D-i they are relabelled by the
    SAME pi as the message (consistent), so coverage is measured against the
    effective post-perm query ids. A relation permutation is a bijection over the
    full base+inverse id set, so a non-degenerate graph yields ~1.0; recorded to
    PROVE the intervention did not strand any query relation. NOT used in any
    decision."""
    dev = test_data.edge_type.device
    q = torch.unique(test_data.target_edge_type)
    if arm == "L0":
        msg_et, q_eff = test_data.edge_type, q
    elif arm == "L4":
        pi = torch.as_tensor(make_role_perm(test_data, R_base, seed)[0],
                             dtype=torch.long, device=dev)
        msg_et, q_eff = pi[test_data.edge_type], q
    elif arm == "L4loose":
        pi = torch.as_tensor(make_loose_perm(R_base, seed)[0],
                             dtype=torch.long, device=dev)
        msg_et, q_eff = pi[test_data.edge_type], q
    elif arm == "D-i":
        pi = torch.as_tensor(make_loose_perm(R_base, seed)[0],
                             dtype=torch.long, device=dev)
        msg_et, q_eff = pi[test_data.edge_type], pi[q]
    else:
        raise ValueError(arm)
    present = set(int(x) for x in torch.unique(msg_et).tolist())
    covered = sum(1 for r in q_eff.tolist() if int(r) in present)
    return round(covered / max(int(q_eff.numel()), 1), 6)


# --------------------------------------------------------------- graph loading ----

def build_graph(name, version, allow_claim=False):
    """Build FLOCK test Data + Ingram filter for an InGram NL/WK graph, exactly as
    the official run_many.py Ingram branch. Claim (OOF) graphs require allow_claim
    (production path only) — else PermissionError (exploration nl-25/wk-25 free)."""
    gkey = f"{name}-{version}"
    if gkey in CLAIM_GRAPH_NAMES and not allow_claim:
        raise PermissionError(
            f"{gkey} is a frozen FLOCK claim (OOF) graph; load only via the "
            "production claim path (validated invocation, allow_claim=True). "
            "Exploration graphs nl-25/wk-25 are unconstrained.")
    ds_cls = {"nl": "NLIngram", "wk": "WKIngram"}[name]
    ctx = {"gpus": [0], "ckpt": FH.CKPT, "epochs": 0, "bpe": "null",
           "dataset": ds_cls, "version": int(version)}
    cfg = FH.util.load_config(FH.CONFIG, context=ctx)
    dataset = FH.util.build_dataset(cfg, device=DEVICE)
    _train, valid, test = (dataset[0].to(DEVICE), dataset[1].to(DEVICE),
                           dataset[2].to(DEVICE))
    fe = torch.cat([valid.edge_index, valid.target_edge_index,
                    test.target_edge_index], dim=1)
    ft = torch.cat([valid.edge_type, valid.target_edge_type,
                    test.target_edge_type])
    filt = Data(edge_index=fe, edge_type=ft, num_nodes=test.num_nodes).to(DEVICE)
    return cfg, test, filt


def load_model(cfg):
    model = FH.Flock(model_cfg=cfg.model)
    state = torch.load(FH.CKPT, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(DEVICE).eval()


# ------------------------------------------------------------------ ladder run ----

def _load_done(jsonl_path):
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[(r["graph"], r["arm"], int(r["seed"]))] = r
    return done


def _append(jsonl_path, rec):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def eval_arm(cfg, model, test, filt, R_base, arm, seed, logger):
    """Build the variant Data for one arm and eval with FLOCK's own ranking (CRN)."""
    if arm == "L0":
        variant, filtv, pstats = test, filt, {"arm": "L0"}
    elif arm == "L4loose":
        pi_np, pstats = make_loose_perm(R_base, seed)
        pi = torch.as_tensor(pi_np, dtype=torch.long, device=DEVICE)
        variant = FH.clone_data_with(test, edge_type=pi[test.edge_type])
        filtv = filt  # query + filter ORIGINAL
    elif arm == "L4":
        pi_np, pstats = make_role_perm(test, R_base, seed)
        pi = torch.as_tensor(pi_np, dtype=torch.long, device=DEVICE)
        variant = FH.clone_data_with(test, edge_type=pi[test.edge_type])
        filtv = filt  # query + filter ORIGINAL
    elif arm == "D-i":
        pi_np, pstats = make_loose_perm(R_base, seed)  # SAME pi as L4loose
        pi = torch.as_tensor(pi_np, dtype=torch.long, device=DEVICE)
        variant = FH.clone_data_with(
            test, edge_type=pi[test.edge_type],
            target_edge_type=pi[test.target_edge_type])
        filtv = FH.clone_data_with(filt, edge_type=pi[filt.edge_type])
    else:
        raise ValueError(arm)
    m = FH.eval_metrics(cfg, model, variant, filtv, DEVICE, logger, seed)
    if variant is not test:
        del variant
    torch.cuda.empty_cache()
    return m, pstats


def run_graph(name, version, model_holder, arms, seeds, jsonl_path,
              allow_claim=False, provenance="fresh"):
    cfg, test, filt = build_graph(name, version, allow_claim=allow_claim)
    R_base = int(test.num_relations) // 2
    logger = FH.util.get_root_logger(file=False)
    gkey = f"{name}-{version}"
    done = _load_done(jsonl_path)
    order = ["L0"] + [a for a in arms if a != "L0"]
    # lazy: only load the checkpoint if some (arm,seed) still needs a GPU eval, so
    # a pure decision/coverage recompute over a full cache touches neither GPU nor .pth.
    need_eval = any((gkey, a, int(s)) not in done for a in order for s in seeds)
    if need_eval and model_holder["m"] is None:
        model_holder["m"] = load_model(cfg)
    model = model_holder["m"]

    # provenance/audit: Pi role stats + inverse-convention census (fresh, cheap)
    _cls, _prof, role_stats = role_partition_base(test, R_base)
    census = {
        "graph": gkey, "n_nodes": int(test.num_nodes),
        "n_msg_edges_base+inv": int(test.edge_index.shape[1]),
        "n_test_queries": int(test.target_edge_index.shape[1]),
        "num_relations": int(test.num_relations), "R_base": R_base,
        "message_edge_type_range": [int(test.edge_type.min()), int(test.edge_type.max())],
        "query_edge_type_range": [int(test.target_edge_type.min()),
                                  int(test.target_edge_type.max())],
        "role_partition": role_stats,
    }

    results = {"L0": {}, "L4loose": {}, "L4": {}, "D-i": {}}
    for arm in order:
        for s in seeds:
            key = (gkey, arm, int(s))
            if key in done:
                results[arm][s] = done[key]["metrics"]
                print(f"[{gkey} {arm} S={s}] (cached:{done[key].get('provenance','?')}) "
                      f"mrr={done[key]['metrics']['mrr']:.6f}", flush=True)
                continue
            t0 = time.time()
            m, pstats = eval_arm(cfg, model, test, filt, R_base, arm, s, logger)
            results[arm][s] = m
            _append(jsonl_path, {
                "graph": gkey, "arm": arm, "seed": int(s), "metrics": m,
                "recovery_vs_L0seed": (None if arm == "L0" or s not in results["L0"]
                                       else m["mrr"] / results["L0"][s]["mrr"]),
                "perm_stats": pstats, "provenance": provenance,
                "build_sec": round(time.time() - t0, 1)})
            print(f"[{gkey} {arm} S={s}] mrr={m['mrr']:.6f} hits@10={m['hits@10']:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # report-only coverage/power flag (review MINOR / §D): applied per arm per seed
    # (graph-only, no eval). Recorded to prove the relabel did not degenerate the task.
    coverage = {arm: {str(s): query_relation_coverage_after_perm(test, R_base, arm, s)
                      for s in seeds} for arm in order}
    census["n_unique_test_query_relations"] = int(
        torch.unique(test.target_edge_type).numel())
    census["query_relation_coverage_after_perm"] = coverage
    census["query_relation_coverage_min"] = min(
        v for a in coverage for v in coverage[a].values())
    return gkey, census, results


# --------------------------------------------------- mechanical decision rules ----

def decide(results, seeds):
    base = [results["L0"][s]["mrr"] for s in seeds]
    band_med = float(np.median(base))
    band = {"min": float(min(base)), "median": band_med, "max": float(max(base)),
            "spread": float(max(base) - min(base))}
    band_rec = {"lo": band["min"] / band_med, "hi": band["max"] / band_med}

    def med_rec(arm):
        vals = [results[arm][s]["mrr"] / results["L0"][s]["mrr"]
                for s in seeds if s in results.get(arm, {})]
        return (float(np.median(vals)) if vals else None), vals

    l4l_med, l4l_all = med_rec("L4loose")
    l4_med, l4_all = med_rec("L4")
    di_med, di_all = med_rec("D-i")
    lo, hi = band_rec["lo"], band_rec["hi"]

    l4l_collapse = (l4l_med is not None) and (l4l_med < lo - COLLAPSE_MARGIN)
    # HONEST split (review MAJOR fix): distinguish STRICT within-class
    # exchangeability from mere ATTENUATION of the collapse. Claiming the former
    # requires the L4 median to sit INSIDE the L0-L0 invariance band; being merely
    # above L4loose (dissociation) is NOT the same thing and must not be labelled
    # "role_preserved".
    l4_in_band = ((l4_med is not None)
                  and (lo - BAND_TOL <= l4_med <= hi + BAND_TOL))
    l4_attenuates = (l4_med is not None and l4l_med is not None
                     and (l4_med - l4l_med) >= DISSOC_MARGIN)
    di_in_band = (di_med is not None) and (lo - BAND_TOL <= di_med <= hi + BAND_TOL)

    if l4_med is None or l4l_med is None:
        l4_interp = "L4/L4loose not both available"
    elif l4_in_band:
        l4_interp = (f"L4 median recovery {l4_med:.3f} lies WITHIN the L0-L0 "
                     f"invariance band [{lo - BAND_TOL:.3f}, {hi + BAND_TOL:.3f}] "
                     "-> consistent with strict within-degree-profile-class "
                     "exchangeability.")
    elif l4_attenuates:
        l4_interp = (f"L4 attenuates collapse (median recovery {l4_med:.3f}, "
                     f"vs L4loose {l4l_med:.3f}) but is BELOW the invariance band "
                     f"[{lo - BAND_TOL:.3f}, {hi + BAND_TOL:.3f}] -> degree-profile "
                     "Pi is a COARSE role proxy (parallels the C-6 "
                     "boundary), NOT full within-class exchangeability.")
    else:
        l4_interp = (f"L4 median recovery {l4_med:.3f} neither reaches the "
                     f"invariance band nor clears the dissociation margin "
                     f"({DISSOC_MARGIN}) over L4loose ({l4l_med:.3f}).")

    return {
        "protocol": "MECHANICAL thresholds only",
        "L0_L0_band_mrr": band,
        "L0_L0_recovery_band": band_rec,
        "recovery_per_seed": {"L4loose": l4l_all, "L4": l4_all, "D-i": di_all},
        "L4loose_median_recovery": l4l_med,
        "L4_median_recovery": l4_med,
        "D-i_median_recovery": di_med,
        "L4_minus_L4loose": (None if (l4_med is None or l4l_med is None)
                             else l4_med - l4l_med),
        "constants": {"COLLAPSE_MARGIN": COLLAPSE_MARGIN, "BAND_TOL": BAND_TOL,
                      "DISSOC_MARGIN": DISSOC_MARGIN},
        "L4loose_collapses": bool(l4l_collapse),
        "L4_in_invariance_band": bool(l4_in_band),
        "L4_attenuates_collapse": bool(l4_attenuates),
        "l4_interpretation": l4_interp,
        "D-i_in_band": bool(di_in_band),
        "matched_pair_dissociation": bool(
            l4l_collapse and l4_attenuates and di_in_band),
    }


# --------------------------------------------------------------- self-tests ----

def _validator_selftest():
    """Verify the freeze validator refuses weakened claim invocations and that the
    claim-graph load guard raises PermissionError off the production path."""
    from argparse import Namespace
    F = FLOCK_CLAIM_FROZEN
    good = Namespace(claim_graphs="nl-50,nl-75,nl-100,wk-75", seeds=5,
                     ckpt="/x/flock_entity.pth", limit=0)
    checks = []

    def expect_exit(tag, ns, arms):
        try:
            validate_flock_claim_invocation(ns, arms)
            checks.append((tag, "FAIL: no SystemExit"))
        except SystemExit:
            checks.append((tag, "OK: refused (SystemExit)"))

    # baseline: exact frozen invocation is accepted as production
    prod = validate_flock_claim_invocation(good, ["L4", "L4loose", "D-i"])
    checks.append(("exact_frozen->production",
                   "OK" if prod == "production" else f"FAIL:{prod}"))
    # weakened: subset graphs
    expect_exit("subset_graphs(nl-50 only)",
                Namespace(**{**vars(good), "claim_graphs": "nl-50"}),
                ["L4", "L4loose", "D-i"])
    # weakened: wrong seeds
    expect_exit("wrong_seeds(3)",
                Namespace(**{**vars(good), "seeds": 3}), ["L4", "L4loose", "D-i"])
    # weakened: --limit set
    expect_exit("limit_set",
                Namespace(**{**vars(good), "limit": 200}), ["L4", "L4loose", "D-i"])
    # weakened: wrong ckpt
    expect_exit("wrong_ckpt",
                Namespace(**{**vars(good), "ckpt": "/x/other.pth"}),
                ["L4", "L4loose", "D-i"])
    # weakened: wrong arms
    expect_exit("wrong_arms(L1,L5)",
                good, ["L1", "L5"])
    # exploration graph exercising claim path -> 'smoke', never evidence
    sm = validate_flock_claim_invocation(
        Namespace(**{**vars(good), "claim_graphs": "nl-25"}), ["L4", "L4loose", "D-i"])
    checks.append(("explore_nl-25->smoke", "OK" if sm == "smoke" else f"FAIL:{sm}"))
    # no claim graphs -> None
    none = validate_flock_claim_invocation(
        Namespace(**{**vars(good), "claim_graphs": ""}), ["L4", "L4loose", "D-i"])
    checks.append(("no_claim->None", "OK" if none is None else f"FAIL:{none}"))
    # load guard: claim graph off production path -> PermissionError
    try:
        build_graph("nl", "50", allow_claim=False)
        checks.append(("load_guard_nl-50", "FAIL: no PermissionError"))
    except PermissionError:
        checks.append(("load_guard_nl-50", "OK: PermissionError"))
    except Exception as e:  # any pre-guard failure is a test failure
        checks.append(("load_guard_nl-50", f"FAIL: {type(e).__name__}:{e}"))
    # exploration graph is NOT guarded at the permission layer (would proceed to load)
    guard_ok = True
    try:
        build_graph("nl", "25", allow_claim=False)  # would build; heavy, so just check no PermissionError before load
    except PermissionError:
        guard_ok = False
    except Exception:
        guard_ok = True  # a non-permission error means the guard let it through (expected)
    checks.append(("explore_nl-25_not_guarded", "OK" if guard_ok else "FAIL: guarded"))

    passed = all(v.startswith("OK") for _, v in checks)
    return {"validator_selftest_pass": bool(passed), "checks": dict(checks)}


def _pi_quickcheck(name, version):
    """Fast Pi-role census on an exploration graph (no eval)."""
    cfg, test, filt = build_graph(name, version, allow_claim=False)
    R_base = int(test.num_relations) // 2
    classes, prof, stats = role_partition_base(test, R_base)
    pi_role, rs = make_role_perm(test, R_base, 0)
    pi_loose, ls = make_loose_perm(R_base, 0)
    # validity: both are permutations; L4 moves only within-class base relations
    ok_role_perm = len(np.unique(pi_role)) == 2 * R_base
    ok_pairing = bool((pi_role[R_base:] == pi_role[:R_base] + R_base).all())
    within_class = True
    prof_np = prof.numpy()
    for r in range(R_base):
        if not np.array_equal(prof_np[pi_role[r]], prof_np[r]):
            within_class = False
            break
    return {"graph": f"{name}-{version}", "R_base": R_base, **stats,
            "L4_pi_is_permutation": bool(ok_role_perm),
            "L4_pairing_preserved": ok_pairing,
            "L4_moves_within_role_class_only": bool(within_class),
            "L4_sigma_moves": int(rs["sigma_moves"]),
            "L4loose_sigma_moves": int(ls["sigma_moves"])}


# --------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS_ALL))
    ap.add_argument("--seeds", type=int, default=3,
                    help="seed count; claim mode requires 5. pre-freeze uses 3")
    ap.add_argument("--explore-graphs", default="nl-25",
                    help="exploration graphs (unconstrained), e.g. 'nl-25'")
    ap.add_argument("--claim-graphs", default="",
                    help="OOF claim graphs 'nl-50,nl-75,nl-100,wk-75' (frozen)")
    ap.add_argument("--ckpt", default=FH.CKPT)
    ap.add_argument("--limit", type=int, default=0,
                    help="pre-freeze speed subsample; forbidden in claim mode")
    ap.add_argument("--run-id", default="FLOCK_LADDER_NL25")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reuse-smoke", default="",
                    help="path to flock_smoke_nl25.json: pre-seed nl-25 "
                         "L0/L4loose/D-i (identical protocol) to skip recompute")
    ap.add_argument("--selftest-validator", action="store_true")
    ap.add_argument("--pi-quickcheck", default="",
                    help="e.g. 'nl-25' -> print Pi role census and exit (no eval)")
    args = ap.parse_args()

    if args.selftest_validator:
        print(json.dumps(_validator_selftest(), indent=2))
        return
    if args.pi_quickcheck:
        n, v = args.pi_quickcheck.split("-")
        print(json.dumps(_pi_quickcheck(n, v), indent=2))
        return

    arms = [a for a in args.arms.split(",") if a]
    claim_mode = validate_flock_claim_invocation(args, arms)
    seeds = list(range(args.seeds))
    out_dir = args.out_dir or os.path.join(_PILOT, "runs", args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "flock_ladder_results.jsonl")

    # optional pre-seed of nl-25 L0/L4loose/D-i from the validated smoke (identical
    # protocol/code/perm): L0<-base_L0, L4loose<-D_ii_L4loose_role_break, D-i<-D_i_*
    if args.reuse_smoke and os.path.exists(args.reuse_smoke) and not os.path.exists(jsonl):
        sm = json.load(open(args.reuse_smoke))
        mp = {"base_L0": "L0", "D_ii_L4loose_role_break": "L4loose",
              "D_i_equivariance_control": "D-i"}
        for sec, arm in mp.items():
            for s, m in sm.get(sec, {}).items():
                _append(jsonl, {"graph": "nl-25", "arm": arm, "seed": int(s),
                                "metrics": m, "provenance": "reused_smoke_identical_protocol"})
        print(f"[reuse-smoke] pre-seeded nl-25 L0/L4loose/D-i from {args.reuse_smoke}",
              flush=True)

    meta = {"run": args.run_id, "backbone": "FLOCK", "ckpt": args.ckpt,
            "arms": arms, "seeds": args.seeds, "claim_mode": claim_mode,
            "explore_graphs": args.explore_graphs, "claim_graphs": args.claim_graphs,
            "limit": args.limit, "device": str(DEVICE),
            "protocol": "FLOCK run.test filtered ranking; matched control = "
                        "message edge_type relation permutation (query+filter fixed "
                        "for L4/L4loose; consistent for D-i); CRN per eval",
            "frozen": {k: (sorted(v) if isinstance(v, set) else v)
                       for k, v in FLOCK_CLAIM_FROZEN.items()}}
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=2)

    model_holder = {"m": None}
    summary = {"status": ("EXPLORATORY_PREFREEZE_VALIDATION_NONCLAIM"
                          if claim_mode != "production" else "PRODUCTION_CLAIM"),
               "meta": meta, "validator_selftest": _validator_selftest(),
               "graphs": {}}

    graph_specs = []
    for g in [x for x in args.explore_graphs.split(",") if x]:
        n, v = g.split("-")
        graph_specs.append((n, v, False))
    for g in [x for x in args.claim_graphs.split(",") if x]:
        n, v = g.split("-")
        graph_specs.append((n, v, claim_mode == "production"))

    for (n, v, allow_claim) in graph_specs:
        gkey, census, results = run_graph(
            n, v, model_holder, arms, seeds, jsonl, allow_claim=allow_claim,
            provenance=("production" if allow_claim else "fresh_explore"))
        dec = decide(results, seeds)
        summary["graphs"][gkey] = {
            "census": census,
            "base_mrr_seeds": {str(s): results["L0"][s]["mrr"] for s in seeds},
            "arm_mrr_seeds": {arm: {str(s): results[arm][s]["mrr"] for s in seeds
                                    if s in results[arm]} for arm in arms},
            "decision": dec}
        json.dump(summary, open(os.path.join(_PILOT, "flock_ladder_summary.json"), "w"),
                  indent=2)

    summary["note"] = ("Mechanical thresholds only. Exploration and "
                       "--limit rows are never claim evidence.")
    json.dump(summary, open(os.path.join(_PILOT, "flock_ladder_summary.json"), "w"),
              indent=2)
    json.dump(summary, open(os.path.join(out_dir, "flock_ladder_summary.json"), "w"),
              indent=2)
    print("[done] flock_ladder ->", os.path.join(_PILOT, "flock_ladder_summary.json"),
          flush=True)


if __name__ == "__main__":
    main()
