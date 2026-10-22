"""native_perturbation.py - the native coherent-perturbation factorial.

Preregistered in refine-logs/PREREG_NATIVE_PERTURBATION_20260903.md, frozen
before this file was executed.

Every claim-bearing arm in the study edits the relation-graph argument alone,
so the study has no scale on which to read the size of that effect. This
experiment supplies one: perturb the entity data coherently, leave the
questions untouched, let the pipeline rebuild the relation graph from the
perturbed data, and sweep the dose.

The construction rests on a verified layout: the inference graph's first half
holds base edges with types in [0,R), and the second half is exactly that half
flipped with type +R.

  Production: CUDA_VISIBLE_DEVICES=<id> python native_perturbation.py
  Selftest (no GPU, no model): python native_perturbation.py --selftest
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import pilot_harness as PH
from ultra.tasks import build_relation_graph

RUN_ID = "NATIVE_PERTURBATION_prereg_20260903"
RUN_ID_CROSS = "NATIVE_PERTURBATION_CROSS_prereg_20260903_A1"
GRAPHS = ("v1", "v2", "v3", "v4")          # FB15k-237 inductive, in-family only
DOSES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2, 3, 4)
MISALIGNED_DOSE = 0.5                       # the fourth cell, descriptive only
LLOOSE_CEILING = 0.186                      # frozen: max archived in-family L4loose median


class CertificateError(AssertionError):
    """A frozen certificate failed. Never retried, never silently repaired."""


def split_halves(data):
    """(base_ei, base_et, R) with the layout certified."""
    ei, et = data.edge_index.cpu(), data.edge_type.cpu()
    m = ei.shape[1]
    R = int(data.num_relations) // 2
    if m % 2:
        raise CertificateError(f"edge count {m} is odd; the base/inverse layout does not hold")
    h = m // 2
    if not bool((et[:h] < R).all()) or not bool((et[h:] >= R).all()):
        raise CertificateError("the first half is not the base half")
    if not torch.equal(et[h:], et[:h] + R) or not torch.equal(ei[:, h:], ei[:, :h].flip(0)):
        raise CertificateError("the second half is not the flipped base half with type +R")
    return ei[:, :h].clone(), et[:h].clone(), R


def perturb_base(base_ei, base_et, q, seed):
    """Redraw the tails of a fraction q of base edges, within their own relation type.

    Each relation keeps its edge count and its head multiset; what changes is
    which head connects to which tail.
    """
    rng = np.random.default_rng(seed)
    ei = base_ei.clone()
    et = base_et.numpy()
    n_moved = 0
    for r in np.unique(et):
        idx = np.nonzero(et == r)[0]
        if idx.size < 2:
            continue
        k = int(round(q * idx.size))
        if k < 2:
            continue
        chosen = rng.choice(idx, size=k, replace=False)
        tails = ei[1, chosen].numpy()
        ei[1, chosen] = torch.as_tensor(rng.permutation(tails), dtype=torch.long)
        n_moved += k
    return ei, int(n_moved)


def perturb_cross(base_ei, base_et, q, seed):
    """Amendment 1: redraw tails ACROSS relation types.

    Each relation still keeps its edge count and its head multiset, but which
    entities are tails of which relation changes, so the tail-incidence sets
    move and the natively rebuilt relation graph moves with them. The
    within-type variant above provably does not: see amendment 1.
    """
    rng = np.random.default_rng(seed)
    ei = base_ei.clone()
    m = ei.shape[1]
    k = int(round(q * m))
    if k < 2:
        return ei, 0
    chosen = rng.choice(m, size=k, replace=False)
    tails = ei[1, chosen].numpy()
    ei[1, chosen] = torch.as_tensor(rng.permutation(tails), dtype=torch.long)
    return ei, int(k)


def rg_jaccard(rg_a, rg_b):
    """Typed-edge-set Jaccard between two relation graphs."""
    def es(rg):
        a = np.stack([rg.edge_index[0].cpu().numpy(), rg.edge_index[1].cpu().numpy(),
                      rg.edge_type.cpu().numpy()])
        return set(map(tuple, a.T.tolist()))
    A, B = es(rg_a), es(rg_b)
    return len(A & B) / len(A | B)


def rebuild_native(data, new_base_ei, base_et, R):
    """Reassemble the full graph from a perturbed base half and rebuild natively."""
    d = data.clone().cpu()
    ei = torch.cat([new_base_ei, new_base_ei.flip(0)], dim=1)
    et = torch.cat([base_et, base_et + R])
    d.edge_index, d.edge_type = ei, et
    if hasattr(d, "relation_graph"):
        del d.relation_graph
    return build_relation_graph(d)


def canonical_typed_edges(ei, et):
    a = np.stack([ei[0].cpu().numpy(), ei[1].cpu().numpy(), et.cpu().numpy()])
    return a[:, np.lexsort((a[2], a[1], a[0]))]


def certify(data, d_new, base_et, R, q):
    """C-NP-1 well-formed, C-NP-2 identity at dose zero, C-NP-3 questions untouched."""
    ei, et = d_new.edge_index.cpu(), d_new.edge_type.cpu()
    h = ei.shape[1] // 2
    if not torch.equal(ei[:, h:], ei[:, :h].flip(0)) or not torch.equal(et[h:], et[:h] + R):
        raise CertificateError("C-NP-1: the inverse half is not the flipped base half")
    if not torch.equal(torch.bincount(et[:h], minlength=R), torch.bincount(base_et, minlength=R)):
        raise CertificateError("C-NP-1: per-relation edge counts changed")

    if q == 0.0:
        want = canonical_typed_edges(data.relation_graph.edge_index, data.relation_graph.edge_type)
        got = canonical_typed_edges(d_new.relation_graph.edge_index, d_new.relation_graph.edge_type)
        if want.shape != got.shape or not np.array_equal(want, got):
            raise CertificateError(
                "C-NP-2: at dose zero the rebuilt relation graph differs from the original")

    if not torch.equal(d_new.target_edge_index.cpu(), data.target_edge_index.cpu()) \
       or not torch.equal(d_new.target_edge_type.cpu(), data.target_edge_type.cpu()):
        raise CertificateError("C-NP-3: the target queries changed")

    jac = rg_jaccard(data.relation_graph, d_new.relation_graph)
    if q > 0.0 and jac >= 1.0:
        raise CertificateError(
            f"C-NP-4: at dose {q} the rebuilt relation graph is identical to the "
            f"original, so this perturbation does not perturb the channel under test")
    return jac


def l4loose_on(d, seed):
    """The channel-only desynchronization, applied to an already-built relation graph."""
    rg = d.relation_graph
    n = int(rg.num_nodes)
    p = torch.as_tensor(np.random.default_rng(10_000 + seed).permutation(n), dtype=torch.long,
                        device=rg.edge_index.device)
    new_ei = torch.stack([p[rg.edge_index[0]], p[rg.edge_index[1]]])
    return PH.rg_clone_into(d, new_ei, rg.edge_type)


def run_graph(model, version, rows, mode="cross"):
    import tunit_common as TC
    data, filtered = PH.load_inductive("fb", version)
    base_ei, base_et, R = split_halves(data)
    base = TC.collect_ranks(model, data, filtered)
    print(f"[fb-{version}] R_base={R} base_edges={base_ei.shape[1]} "
          f"base MRR={base['mrr']:.6f}", flush=True)

    for q in DOSES:
        for seed in SEEDS:
            t0 = time.time()
            pfn = perturb_cross if mode == "cross" else perturb_base
            new_ei, n_moved = pfn(base_ei, base_et, q, seed)
            d_new = rebuild_native(data, new_ei, base_et, R)
            jac = certify(data, d_new, base_et, R, q)
            d_dev = d_new.to(PH.DEVICE)
            res = TC.collect_ranks(model, d_dev, filtered)
            row = {"run_id": RUN_ID, "mode": mode, "graph": f"fb-{version}", "arm": "aligned",
                   "dose": q, "seed": int(seed), "n_tails_redrawn": n_moved, "rg_jaccard": jac,
                   "base_mrr": float(base["mrr"]), "mrr": float(res["mrr"]),
                   "recovery": float(res["mrr"]) / float(base["mrr"]),
                   "hits@10": float(res["hits@10"]), "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print("ROW " + json.dumps(row), flush=True)

            if q == MISALIGNED_DOSE:
                t0 = time.time()
                res2 = TC.collect_ranks(model, l4loose_on(d_dev, seed), filtered)
                row2 = dict(row, arm="misaligned", mrr=float(res2["mrr"]),
                            recovery=float(res2["mrr"]) / float(base["mrr"]),
                            hits__10=None, seconds=round(time.time() - t0, 1))
                row2["hits@10"] = float(res2["hits@10"])
                row2.pop("hits__10")
                rows.append(row2)
                print("ROW " + json.dumps(row2), flush=True)


def selftest():
    """Construction only: no model, no GPU, no dataset."""
    R = 5
    base_ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [10, 11, 12, 13, 14, 15, 16, 17]])
    base_et = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    ei0, n0 = perturb_base(base_ei, base_et, 0.0, 0)
    assert torch.equal(ei0, base_ei) and n0 == 0, "dose zero is not a no-op"

    ei1, n1 = perturb_base(base_ei, base_et, 1.0, 1)
    assert n1 == 8, n1
    assert torch.equal(ei1[0], base_ei[0]), "heads must not move"
    for r in (0, 1):
        m = base_et.numpy() == r
        assert sorted(ei1[1][m].tolist()) == sorted(base_ei[1][m].tolist()), \
            "tails must stay inside their own relation type"
    assert not torch.equal(ei1, base_ei), "full dose changed nothing"

    # C-NP-1 catches a broken inverse half
    class D:
        pass
    d = D()
    good_ei = torch.cat([ei1, ei1.flip(0)], dim=1)
    good_et = torch.cat([base_et, base_et + R])
    d.edge_index, d.edge_type = good_ei, good_et
    d.target_edge_index, d.target_edge_type = base_ei, base_et
    src = D()
    src.target_edge_index, src.target_edge_type = base_ei, base_et

    class RG:
        def __init__(self, ei, et):
            self.edge_index, self.edge_type = ei, et
    same = RG(torch.tensor([[0, 1], [1, 2]]), torch.tensor([0, 1]))
    moved = RG(torch.tensor([[0, 1], [1, 3]]), torch.tensor([0, 1]))
    src.relation_graph, d.relation_graph = same, moved
    assert certify(src, d, base_et, R, 0.5) < 1.0

    # C-NP-4 must fire when the perturbation leaves the relation graph untouched
    d.relation_graph = RG(same.edge_index.clone(), same.edge_type.clone())
    try:
        certify(src, d, base_et, R, 0.5)
        raise SystemExit("selftest failed: C-NP-4 accepted an unchanged relation graph")
    except CertificateError:
        pass
    # and must not fire at dose zero, where the graph is supposed to be unchanged
    assert certify(src, d, base_et, R, 0.0) == 1.0

    d.relation_graph = moved
    d.edge_type = torch.cat([base_et, base_et + R + 1])
    try:
        certify(src, d, base_et, R, 0.5)
        raise SystemExit("selftest failed: C-NP-1 accepted a broken inverse half")
    except CertificateError:
        pass
    print("selftest OK: dose-zero no-op, head fixity, within-type tails, C-NP-1, C-NP-4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fb-v1 only, one seed per dose")
    ap.add_argument("--mode", choices=("within", "cross"), default="cross",
                    help="cross is the amendment-1 construction; within is the original")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    global GRAPHS, SEEDS
    if args.smoke:
        GRAPHS, SEEDS = ("v1",), (0,)

    rid = RUN_ID_CROSS if args.mode == "cross" else RUN_ID
    out_dir = os.path.join(PH.PILOT_DIR, "runs", rid + ("_SMOKE" if args.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)
    model = PH.load_model()
    rows = []
    for v in GRAPHS:
        run_graph(model, v, rows, args.mode)

    al = [r for r in rows if r["arm"] == "aligned"]
    curve = {}
    for q in DOSES:
        per_graph = []
        for g in sorted({r["graph"] for r in al}):
            v = [r["recovery"] for r in al if r["graph"] == g and r["dose"] == q]
            if v:
                per_graph.append(float(np.median(v)))
        if per_graph:
            curve[str(q)] = {"per_graph_median": per_graph,
                             "median_over_graphs": float(np.median(per_graph))}
    reached = [q for q in DOSES
               if str(q) in curve and curve[str(q)]["median_over_graphs"] <= LLOOSE_CEILING]
    summary = {"run_id": rid, "mode": args.mode, "smoke": bool(args.smoke), "n_rows": len(rows),
               "ceiling": LLOOSE_CEILING, "aligned_curve": curve,
               "q_star": (min(reached) if reached else None),
               "outcome": ("B" if reached and min(reached) <= 0.5 else
                           "C" if reached else "A"),
               "misaligned_at_%.2f" % MISALIGNED_DOSE: {
                   r["graph"]: r["recovery"] for r in rows if r["arm"] == "misaligned"},
               "numpy": np.__version__, "torch": torch.__version__}
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
