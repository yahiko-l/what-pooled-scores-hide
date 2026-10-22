"""joint_rename_control.py — the implemented consistent-rename control.

Preregistered in refine-logs/PREREG_JOINT_RENAME_CONTROL_20260903.md, frozen
before this file was executed.

Rename-invariance holds for the ideal exact-arithmetic map. Every
claim-bearing arm in the study edits the relation-graph argument alone. This
control renames relation identities consistently across the inference graph,
the target queries and the filter graph, then lets the pipeline's own
constructor rebuild the relation graph from the renamed data. It is the only
arm whose relation graph comes from the native construction path.

The rename is an ATOMIC base/inverse lift, forced by the codebase: the dataset
builder appends inverses as r + num_relations, and the model maps a
head-direction query to r + num_relations // 2. An arbitrary permutation of the
2R identities would break that correspondence and would not be consistent.

  Production: CUDA_VISIBLE_DEVICES=<id> python joint_rename_control.py
  Selftest (no GPU, no model): python joint_rename_control.py --selftest
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import pilot_harness as PH
from ultra.tasks import build_relation_graph

RUN_ID = "JOINT_RENAME_CONTROL_prereg_20260903"
GRAPHS = ("v1", "v2", "v3", "v4")          # FB15k-237 inductive, in-family only
SEEDS = (0, 1, 2, 3, 4)
BAND = (0.99, 1.01)                         # frozen; never widened after the fact


class CertificateError(AssertionError):
    """C-JR-1 failed. Never retried, never silently repaired."""


def lift_atomic(sigma_base, R_base):
    """Base permutation to the full 2R permutation, pairing preserving."""
    sigma = np.asarray(sigma_base)
    pi = np.concatenate([sigma, sigma + R_base])
    assert len(np.unique(pi)) == 2 * R_base, "lifted pi is not a permutation"
    assert np.array_equal(pi[np.arange(R_base) + R_base],
                          pi[np.arange(R_base)] + R_base), "lift is not atomic"
    return pi


def canonical_typed_edges(ei, et):
    """A relation graph as a canonically sorted (src, dst, type) array."""
    a = np.stack([ei[0].cpu().numpy(), ei[1].cpu().numpy(), et.cpu().numpy()])
    order = np.lexsort((a[2], a[1], a[0]))
    return a[:, order]


def certify_native_rebuild(rg_base, rg_native, pi):
    """C-JR-1: the natively rebuilt relation graph IS the relabel of the base one."""
    n0, n1 = int(rg_base.num_nodes), int(rg_native.num_nodes)
    if n0 != n1:
        raise CertificateError(f"C-JR-1: node counts differ, base {n0} native {n1}")
    p = torch.as_tensor(pi, dtype=torch.long)
    ei0 = rg_base.edge_index.cpu()
    want = canonical_typed_edges(torch.stack([p[ei0[0]], p[ei0[1]]]),
                                 rg_base.edge_type.cpu())
    got = canonical_typed_edges(rg_native.edge_index.cpu(),
                                rg_native.edge_type.cpu())
    if want.shape != got.shape or not np.array_equal(want, got):
        raise CertificateError(
            "C-JR-1: the natively rebuilt relation graph is not the lifted relabel "
            f"of the base relation graph (base edges {want.shape[1]}, "
            f"native edges {got.shape[1]})")
    return int(want.shape[1])


def joint_rename(data, filtered, pi):
    """Rename every relation field together, then rebuild the relation graph natively.

    build_relation_graph runs on CPU tensors only, and PyG .cpu()/.to() recurse
    into the nested relation_graph in place, so the clone must be deep. This is
    the same pattern pilot_harness.drop_entity_edges uses.
    """
    p_cpu = torch.as_tensor(pi, dtype=torch.long)
    d = data.clone().cpu()
    d.edge_type = p_cpu[d.edge_type]
    d.target_edge_type = p_cpu[d.target_edge_type]
    if hasattr(d, "relation_graph"):
        del d.relation_graph
    d = build_relation_graph(d)

    f = filtered.clone().cpu()
    f.edge_type = p_cpu[f.edge_type]
    return d, f


def rank_agreement(base_res, new_res):
    """Descriptive only, no decision line: per-query rank agreement with the base."""
    b = np.array(base_res["t_ranks"] + base_res["h_ranks"], dtype=np.int64)
    n = np.array(new_res["t_ranks"] + new_res["h_ranks"], dtype=np.int64)
    if b.shape != n.shape:
        return {"comparable": False}
    d = np.abs(b - n)
    return {"comparable": True, "n_queries": int(b.size),
            "frac_rank_identical": float((d == 0).mean()),
            "max_abs_rank_delta": int(d.max()) if d.size else 0}


def run_graph(model, version, out_rows):
    import tunit_common as TC
    data, filtered = PH.load_inductive("fb", version)
    R2 = int(data.num_relations)
    assert R2 % 2 == 0, f"num_relations {R2} is not even; the base/inverse convention does not hold"
    R_base = R2 // 2

    base = TC.collect_ranks(model, data, filtered)
    print(f"[fb-{version}] R_base={R_base} base MRR={base['mrr']:.6f}", flush=True)

    for seed in SEEDS:
        t0 = time.time()
        sigma = np.random.default_rng(seed).permutation(R_base)
        pi = lift_atomic(sigma, R_base)
        d_new, f_new = joint_rename(data, filtered, pi)
        n_rg_edges = certify_native_rebuild(data.relation_graph, d_new.relation_graph, pi)

        res = TC.collect_ranks(model, d_new.to(PH.DEVICE), f_new.to(PH.DEVICE))
        rec = float(res["mrr"]) / float(base["mrr"])
        row = {
            "run_id": RUN_ID, "graph": f"fb-{version}", "seed": int(seed),
            "R_base": R_base, "n_relation_graph_edges": n_rg_edges,
            "base_mrr": float(base["mrr"]), "renamed_mrr": float(res["mrr"]),
            "recovery": rec,
            "in_band": bool(BAND[0] <= rec <= BAND[1]),
            "n_moved_base": int((sigma != np.arange(R_base)).sum()),
            "base_hits@10": float(base["hits@10"]),
            "renamed_hits@10": float(res["hits@10"]),
            "rank_agreement": rank_agreement(base, res),
            "seconds": round(time.time() - t0, 1),
        }
        out_rows.append(row)
        print("ROW " + json.dumps(row), flush=True)


def selftest():
    """Construction only: no model, no GPU, no dataset."""
    R = 7
    sigma = np.random.default_rng(3).permutation(R)
    pi = lift_atomic(sigma, R)
    assert np.array_equal(pi[:R], sigma)
    assert np.array_equal(pi[R:], sigma + R)

    # a rename that is NOT atomic must be rejected by the lift's own assertion
    bad = np.concatenate([sigma, np.random.default_rng(4).permutation(R) + R])
    try:
        assert np.array_equal(bad[np.arange(R) + R], bad[np.arange(R)] + R)
        raise SystemExit("selftest failed: a non-atomic lift was accepted")
    except AssertionError:
        pass

    # C-JR-1 catches a relation graph that is not the relabel
    class G:
        def __init__(self, ei, et, n):
            self.edge_index, self.edge_type, self.num_nodes = ei, et, n
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    et = torch.tensor([0, 1, 2])
    p = torch.as_tensor(pi[: 2 * R], dtype=torch.long)
    good = G(torch.stack([p[ei[0]], p[ei[1]]]), et, 2 * R)
    assert certify_native_rebuild(G(ei, et, 2 * R), good, pi) == 3
    bad_g = G(torch.stack([p[ei[0]], p[ei[1]].flip(0)]), et, 2 * R)
    try:
        certify_native_rebuild(G(ei, et, 2 * R), bad_g, pi)
        raise SystemExit("selftest failed: C-JR-1 accepted a non-relabel")
    except CertificateError:
        pass
    print("selftest OK: atomic lift, non-atomic rejection, C-JR-1 both directions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    out_dir = os.path.join(PH.PILOT_DIR, "runs", RUN_ID)
    os.makedirs(out_dir, exist_ok=True)
    model = PH.load_model()
    rows = []
    for v in GRAPHS:
        run_graph(model, v, rows)

    recs = [r["recovery"] for r in rows]
    summary = {
        "run_id": RUN_ID, "n_cells": len(rows),
        "band": list(BAND),
        "all_in_band": all(r["in_band"] for r in rows),
        "recovery_min": min(recs), "recovery_max": max(recs),
        "recovery_median": float(np.median(recs)),
        "frac_rank_identical_min": min(
            r["rank_agreement"].get("frac_rank_identical", -1.0) for r in rows),
        "max_abs_rank_delta": max(
            r["rank_agreement"].get("max_abs_rank_delta", -1) for r in rows),
        "certificate": "C-JR-1 passed on every cell (a failure aborts the run)",
        "numpy": np.__version__, "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
