"""FLOCK under the documented predicate merges of the held-out Wikidata graphs (amendment A3).

The merged graphs, the targets each evaluation scores, the strata and the displacement tiers are
those of the ULTRA cells of amendment A2 (www_schema_merge.py, MERGE_SPEC_A2.json, rule set yago).
--export builds them with www_schema_merge.build_cell in a process that loads ULTRA's dataset
classes and saves them to the run directory, and certifies them against the structure rows of the A2
ULTRA run. A FLOCK process never imports the ULTRA stack: it loads each graph the way FLOCK's runner
does (flock_ladder.build_graph), certifies that every tensor equals the exported original, and
replaces the message edges, the relation count, the targets and the filter with the exported merged
ones. FLOCK has no relation graph, so nothing else is rebuilt.

A unit is (graph, seed). It evaluates the original and the merged graph at common random numbers,
seed s for both, on the subsets pooled, affected and the affected tiers high, mid and low. The rows
of a unit are appended together after the unit succeeds; the attempt ledger records each start,
success and failure.

  python www_flock_merge.py --selftest                       CPU, no data
  python www_flock_merge.py --export                          CPU, ULTRA's dataset classes
  python www_flock_merge.py --certify                         CPU, FLOCK's dataset classes
  CUDA_VISIBLE_DEVICES=<id> python www_flock_merge.py --smoke
  CUDA_VISIBLE_DEVICES=<id> python www_flock_merge.py --commit <sha> [--graphs G] [--seeds S] [--resume]
  python www_flock_merge.py --check
  python www_flock_merge.py --analyze
"""
import os

# Read by numpy at import; see www_strata_replication.py.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import matched_pairs as MP  # noqa: E402
import www_gpu  # noqa: E402

CPU_MODES = {"--selftest", "--export", "--certify", "--check", "--analyze", "--verify-commit",
             "-h", "--help"}
if __name__ == "__main__" and not CPU_MODES & set(sys.argv[1:]):
    # The GPU-sharing ruling covers the A1 and A2 runs only; this run uses free GPUs.
    www_gpu.MAX_MEMORY_MIB = 500
    www_gpu.admit()

import torch  # noqa: E402

AMENDMENT = "A3"
TAG = "prereg-www2027-a3"
GRAPHS = ("wk-50", "wk-100")
SEEDS = (0, 1, 2, 3, 4)
RULESET = "yago"
SUBSETS = ("pooled", "affected", "affected_high", "affected_mid", "affected_low")
TIERS = ("high", "mid", "low")
CKPT = MP.CKPTS["flock"][0]
RUN, SMOKE_RUN = "WWW_MERGE_A3_FLOCK", "WWW_MERGE_A3_FLOCK_SMOKE"
RUNS_DIR = os.path.join(MP.PILOT_DIR, "runs")
RUN_DIR = os.path.join(RUNS_DIR, RUN)
RESULTS = os.path.join(RUN_DIR, "flock_merge_results.jsonl")
LEDGER = os.path.join(RUN_DIR, "ATTEMPTS.jsonl")
A2_ULTRA_ROWS = os.path.join(RUNS_DIR, "WWW_MERGE_A2", "www_merge_results.jsonl")
SPEC = os.path.join(MP.REPO_ROOT, "refine-logs", "www2027", "MERGE_SPEC_A2.json")
CUTOFF_UTC = "2026-10-22T12:00:00+00:00"
LOCAL_MIN, LOCAL_RATIO = 0.02, 3.0
SMOKE_TARGETS = 32
ROW_FIELDS = ("run", "mode", "backbone", "commit", "spec", "graph", "ruleset", "ckpt", "kind")
TENSORS = ("edge_index", "edge_type", "target_edge_index", "target_edge_type",
           "filter_edge_index", "filter_edge_type")
# Project modules a production process executes; no git is read on the execution host, so each unit
# records their SHA-256 and --check compares them with the files of the recorded commit.
CODE_FILES = ("www_flock_merge.py", "flock_ladder.py", "flock_harness.py", "matched_pairs.py",
              "matched_construction.py", "www_gpu.py")


def construction_path(gkey):
    return os.path.join(RUN_DIR, f"construction_{gkey}.pt")


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def code_digests():
    out = {}
    for name in CODE_FILES:
        with open(os.path.join(MP.PILOT_DIR, name), "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def subset_masks(labels, tiers, idx):
    lab, tr = labels[idx], tiers[idx]
    masks = {"pooled": np.ones(idx.size, dtype=bool), "affected": lab == "affected"}
    for t in TIERS:
        masks["affected_" + t] = tr == t
    return masks


# ---------------------------------------------------------- export (ULTRA) ----

def _tensors(data, filtered):
    return {"edge_index": data.edge_index.cpu(), "edge_type": data.edge_type.cpu(),
            "target_edge_index": data.target_edge_index.cpu(),
            "target_edge_type": data.target_edge_type.cpu(),
            "num_nodes": int(data.num_nodes), "num_relations": int(data.num_relations),
            "filter_edge_index": filtered.edge_index.cpu(),
            "filter_edge_type": filtered.edge_type.cpu()}


def a2_structure(gkey):
    """The structure row of the A2 ULTRA cell (graph, yago), without the run fields; identical for
    every checkpoint."""
    rows = [{k: v for k, v in r.items() if k not in ROW_FIELDS} for r in read_jsonl(A2_ULTRA_ROWS)
            if r.get("kind") == "structure" and r["graph"] == gkey and r["ruleset"] == RULESET]
    if not rows:
        raise SystemExit(f"no A2 ULTRA structure row for {gkey} in {A2_ULTRA_ROWS}")
    if any(r != rows[0] for r in rows):
        raise MP.CertificateError(f"{gkey}: the A2 structure rows disagree across checkpoints")
    return rows[0]


def export():
    import www_schema_merge as WSM
    spec = WSM.load_spec(SPEC)
    if WSM.spec_amendment(spec) != "A2":
        raise SystemExit(f"{SPEC} is not the A2 specification")
    os.makedirs(RUN_DIR, exist_ok=True)
    for gkey in GRAPHS:
        data, filtered, names, _ = WSM.load_graph(gkey)
        cell = WSM.build_cell(gkey, RULESET, None, data, filtered, names, WSM.spec_groups(spec, gkey))
        st = json.loads(json.dumps(cell["structure"]))
        ref = a2_structure(gkey)
        if st != ref:
            bad = sorted(k for k in set(st) | set(ref) if st.get(k) != ref.get(k))
            raise MP.CertificateError(f"{gkey}: construction differs from the A2 ULTRA run in {bad}")
        torch.save({"graph": gkey, "ruleset": RULESET, "spec": os.path.relpath(SPEC, MP.REPO_ROOT),
                    "structure": st, "original": _tensors(data, filtered),
                    "merged": _tensors(cell["merged"], cell["filter"]),
                    "drop": cell["drop"].tolist(), "keep_merged": cell["keep_merged"].tolist(),
                    "labels": cell["labels"].tolist(), "tiers": cell["tiers"].tolist()},
                   construction_path(gkey))
        print(f"[export] {gkey}: R {st['R']} -> {st['R_merged']}, targets {st['n_targets']}, "
              f"affected {st['strata']['affected']}, tiers {st['tiers']}, dropped {st['n_dropped']}, "
              f"collapsed {st['n_collapsed']}; equal to the A2 ULTRA structure rows "
              f"-> {construction_path(gkey)}", flush=True)


def load_construction(gkey):
    exp = torch.load(construction_path(gkey))
    for k in ("drop", "keep_merged"):
        exp[k] = np.asarray(exp[k], dtype=bool)
    for k in ("labels", "tiers"):
        exp[k] = np.asarray(exp[k], dtype=object).astype(str)
    st = exp["structure"]
    if (int((~exp["drop"]).sum()) != st["n_queries_original"]
            or int(exp["keep_merged"].sum()) != st["n_queries_merged"]
            or exp["merged"]["target_edge_type"].numel() != st["n_queries_merged"]
            or len(exp["labels"]) != st["n_targets"]):
        raise MP.CertificateError(f"{gkey}: construction file is inconsistent with its structure")
    return exp


# ------------------------------------------------------------------ FLOCK ----

def load_flock_graph(gkey):
    import flock_ladder as FL
    name, ver = gkey.split("-")
    return FL.build_graph(name, ver, allow_claim=False)


def certify(gkey, test, filt, exp):
    o = exp["original"]
    have = {"edge_index": test.edge_index, "edge_type": test.edge_type,
            "target_edge_index": test.target_edge_index, "target_edge_type": test.target_edge_type,
            "filter_edge_index": filt.edge_index, "filter_edge_type": filt.edge_type}
    bad = [k for k in TENSORS if not torch.equal(have[k].cpu(), o[k])]
    if int(test.num_nodes) != o["num_nodes"] or int(test.num_relations) != o["num_relations"]:
        bad.append("num_nodes/num_relations")
    if int(filt.num_nodes) != o["num_nodes"]:
        bad.append("filter num_nodes")
    if bad:
        raise MP.CertificateError(f"{gkey}: FLOCK's graph differs from the exported original in {bad}")


def with_targets(data, idx):
    d = copy.copy(data)
    i = torch.as_tensor(idx, dtype=torch.long, device=data.target_edge_type.device)
    d.target_edge_index = data.target_edge_index[:, i]
    d.target_edge_type = data.target_edge_type[i]
    return d


def merged_inputs(test, filt, exp):
    import flock_harness as FH
    from torch_geometric.data import Data
    m, dev = exp["merged"], test.edge_index.device
    merged = FH.clone_data_with(
        test, edge_index=m["edge_index"].to(dev), edge_type=m["edge_type"].to(dev),
        num_relations=torch.as_tensor([m["num_relations"]], dtype=test.num_relations.dtype,
                                      device=test.num_relations.device),
        target_edge_index=m["target_edge_index"].to(dev),
        target_edge_type=m["target_edge_type"].to(dev))
    merged_filter = Data(edge_index=m["filter_edge_index"].to(dev),
                         edge_type=m["filter_edge_type"].to(dev), num_nodes=int(filt.num_nodes))
    return merged, merged_filter


def common(gkey, seed, mode, commit):
    return {"run": RUN if mode == "production" else SMOKE_RUN, "amendment": AMENDMENT,
            "mode": mode, "label": "WWW_MERGE_A3 FLOCK under documented merges" if mode ==
            "production" else MP.SMOKE_LABEL, "backbone": "flock", "ckpt": CKPT, "graph": gkey,
            "ruleset": RULESET, "seed": int(seed), "crn_seed": int(seed), "commit": commit}


def flock_unit(cfg, model, gkey, test, filt, exp, seed, mode, commit, limit=None):
    """Original and merged evaluations of one (graph, seed) unit at CRN seed `seed`."""
    import flock_harness as FH
    import flock_ladder as FL
    logger = FH.util.get_root_logger(file=False)
    if limit is not None:
        # FLOCK's test() logs its metrics at WARNING; a smoke unit must not show them.
        logger.setLevel(logging.ERROR)
    snap = {k: v.clone() for k, v in (("ei", test.edge_index), ("et", test.edge_type),
                                      ("tei", test.target_edge_index), ("tet", test.target_edge_type),
                                      ("fei", filt.edge_index), ("fet", filt.edge_type))}
    keep_o = np.nonzero(~exp["drop"])[0]
    keep_m = np.nonzero(exp["keep_merged"])[0]
    merged, merged_filter = merged_inputs(test, filt, exp)
    if limit is not None:
        keep_o, keep_m = keep_o[:limit], keep_m[:limit]
        merged = with_targets(merged, np.arange(keep_m.size))
    data_o = with_targets(test, keep_o)
    masks_o = subset_masks(exp["labels"], exp["tiers"], keep_o)
    masks_m = subset_masks(exp["labels"], exp["tiers"], keep_m)
    key = common(gkey, seed, mode, commit)
    where = f"{gkey} s{seed}"
    rows = []
    for s in (("pooled",) if limit is not None else SUBSETS):
        row = {**key, "kind": "eval", "subset": s}
        for side, data, mask, flt in (("original", data_o, masks_o[s], filt),
                                       ("merged", merged, masks_m[s], merged_filter)):
            n = int(mask.sum())
            row[f"n_queries_{side}"] = n
            mrr = None if n == 0 else MP.check_mrr(FH.eval_metrics(
                cfg, model, with_targets(data, np.nonzero(mask)[0]), flt, FL.DEVICE, logger,
                seed)["mrr"], f"{where} {s} {side}")
            # A smoke unit is read as status only: its scores are checked, never recorded.
            if limit is None:
                row[f"mrr_{side}"] = mrr
            else:
                row[f"evaluated_{side}"] = mrr is not None
        rows.append(row)
    for k, v in (("ei", test.edge_index), ("et", test.edge_type), ("tei", test.target_edge_index),
                 ("tet", test.target_edge_type), ("fei", filt.edge_index), ("fet", filt.edge_type)):
        if not torch.equal(v, snap[k]):
            raise MP.CertificateError(f"{where}: original tensor {k} mutated")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def done_units(path):
    return {(r["graph"], int(r["seed"])) for r in read_jsonl(path) if r.get("kind") == "unit_done"}


def parse_list(spec, allowed, cast):
    vals = [cast(x) for x in spec.split(",") if x.strip()] if spec else list(allowed)
    outside = [v for v in vals if v not in allowed]
    if outside:
        raise SystemExit(f"{outside}: not among {list(allowed)}")
    return vals


def run(mode, commit, graphs, seeds, resume):
    import flock_ladder as FL
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible: set CUDA_VISIBLE_DEVICES to a free GPU")
    if mode == "production":
        if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise SystemExit("production requires --commit <40-character sha of the tagged code>")
        os.makedirs(RUN_DIR, exist_ok=True)
        out_path = RESULTS
        if os.path.exists(out_path) and not resume:
            raise SystemExit(f"{out_path} exists; pass --resume to continue the missing units")
    else:
        out_dir = os.path.join(RUNS_DIR, SMOKE_RUN)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"smoke_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    done = done_units(out_path) if resume else set()
    digests = code_digests()
    model = None
    for gkey in graphs:
        todo = [s for s in seeds if (gkey, s) not in done]
        if not todo:
            continue
        exp = load_construction(gkey)
        cfg, test, filt = load_flock_graph(gkey)
        certify(gkey, test, filt, exp)
        if model is None:
            model = FL.load_model(cfg)
        for seed in todo:
            www_gpu.between_units(f"{gkey} s{seed}")
            unit = f"{gkey}:s{seed}"
            if mode == "production":
                append_jsonl(LEDGER, {"ts": now_utc(), "unit": unit, "event": "start",
                                      "pid": os.getpid(),
                                      "gpu_index": (www_gpu.current() or {}).get("index")})
            t0 = time.time()
            try:
                rows = flock_unit(cfg, model, gkey, test, filt, exp, seed, mode, commit,
                                  limit=SMOKE_TARGETS if mode == "smoke" else None)
            except Exception as e:  # noqa: BLE001 (recorded in the ledger, then a nonzero exit)
                if mode == "production":
                    append_jsonl(LEDGER, {"ts": now_utc(), "unit": unit, "event": "failure",
                                          "error": repr(e)[:500]})
                print(f"[{mode}] {unit}: FAILED {e!r}", flush=True)
                raise
            rows.append({**common(gkey, seed, mode, commit), "kind": "unit_done",
                         "seconds": round(time.time() - t0, 1), "completed_utc": now_utc(),
                         "code_sha256": digests})
            with open(out_path, "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            if mode == "production":
                append_jsonl(LEDGER, {"ts": now_utc(), "unit": unit, "event": "success",
                                      "rows": len(rows)})
            print(f"[{mode}] {unit}: {len(rows) - 1} subsets in {time.time() - t0:.0f}s", flush=True)
        del test, filt
    print(f"done -> {out_path}")


def certify_all():
    for gkey in GRAPHS:
        exp = load_construction(gkey)
        _, test, filt = load_flock_graph(gkey)
        certify(gkey, test, filt, exp)
        m = exp["merged"]
        if int(m["edge_type"].max()) >= m["num_relations"] or int(m["target_edge_type"].max()) >= \
                m["num_relations"] // 2:
            raise MP.CertificateError(f"{gkey}: merged relation ids out of range")
        print(f"[certify] {gkey}: FLOCK's graph equals the exported original in every tensor; "
              f"merged relations {m['num_relations'] // 2}, merged targets "
              f"{m['target_edge_type'].numel()}", flush=True)


# ------------------------------------------------------------ check, analysis ----

def git_digest(commit, name):
    path = f"idea-stage/pilot/{name}"
    blob = subprocess.run(["git", "-C", MP.REPO_ROOT, "show", f"{commit}:{path}"],
                          capture_output=True, check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def tag_commit():
    return subprocess.run(["git", "-C", MP.REPO_ROOT, "rev-parse", f"{TAG}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def verify_commit(commit):
    """Before launch, on the repository host: the commit is the one the freeze tag names, and the
    digests printed here are the ones the execution host's copies must have."""
    if commit != tag_commit():
        raise SystemExit(f"{commit} is not the commit tagged {TAG} ({tag_commit()})")
    for name in CODE_FILES:
        print(f"{git_digest(commit, name)}  {name}")


def check():
    """Completeness, consistency and provenance of the production rows (run where git is)."""
    rows = read_jsonl(RESULTS)
    problems, notes = [], []
    units = {}
    for r in rows:
        units.setdefault((r["graph"], int(r["seed"])), []).append(r)
    commits = {r["commit"] for r in rows}
    if commits != {tag_commit()}:
        problems.append(f"commits {sorted(map(str, commits))}, tag {TAG} names {tag_commit()}")
    successes = {e["unit"] for e in read_jsonl(LEDGER) if e["event"] == "success"}
    for gkey in GRAPHS:
        st = a2_structure(gkey)
        for seed in SEEDS:
            rs = units.get((gkey, seed), [])
            evals = {r["subset"]: r for r in rs if r["kind"] == "eval"}
            dones = [r for r in rs if r["kind"] == "unit_done"]
            if len(dones) != 1 or set(evals) != set(SUBSETS) or len(evals) != len(rs) - 1:
                problems.append(f"{gkey} s{seed}: {len(dones)} unit_done rows, subsets {sorted(evals)}")
                continue
            if evals["pooled"]["n_queries_original"] != st["n_queries_original"] or \
                    evals["pooled"]["n_queries_merged"] != st["n_queries_merged"]:
                problems.append(f"{gkey} s{seed}: pooled query counts differ from the construction")
            if evals["affected"]["n_queries_original"] != st["strata"]["affected"] - \
                    st["dropped_by_stratum"]["affected"]:
                problems.append(f"{gkey} s{seed}: affected query count differs from the construction")
            for t in TIERS:
                if st["n_dropped"] == 0 and evals["affected_" + t]["n_queries_original"] != \
                        st["tiers"][t]:
                    problems.append(f"{gkey} s{seed}: tier {t} count differs from the construction")
            for name, digest in dones[0]["code_sha256"].items():
                if digest != git_digest(dones[0]["commit"], name):
                    problems.append(f"{gkey} s{seed}: {name} differs from the recorded commit")
            if f"{gkey}:s{seed}" not in successes:
                notes.append(f"{gkey} s{seed}: no ledger success; its rows record completion")
    out = {"n_rows": len(rows), "units": len(units), "commits": sorted(map(str, commits)),
           "problems": problems, "notes": notes, "status": "PASS" if not problems else "FAIL"}
    with open(os.path.join(RUN_DIR, "production_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    if problems:
        raise SystemExit(1)


def completion_times(rows):
    return {(r["graph"], int(r["seed"])): r["completed_utc"] for r in rows
            if r.get("kind") == "unit_done" and r.get("mode") == "production"}


def _summary(vals):
    """Medians over the seeds (or checkpoints) of one subset; scores are None for an empty subset."""
    n = {"n_original": vals[0]["n_original"], "n_merged": vals[0]["n_merged"]}
    if vals[0]["original"] is None:
        return {**n, "original": None, "merged": None, "delta": None, "abs_delta": None,
                "relative": None, "lower": None}
    return {**n, "original": statistics.median(v["original"] for v in vals),
            "merged": statistics.median(v["merged"] for v in vals),
            "delta": statistics.median(v["delta"] for v in vals),
            "abs_delta": statistics.median(abs(v["delta"]) for v in vals),
            "relative": statistics.median(v["relative"] for v in vals),
            "lower": sum(v["delta"] < 0 for v in vals)}


def _entry(r):
    o, m = r["mrr_original"], r["mrr_merged"]
    d = None if o is None or m is None else m - o
    return {"original": o, "merged": m, "delta": d,
            "relative": None if d is None or not o else d / o,
            "n_original": r["n_queries_original"], "n_merged": r["n_queries_merged"]}


def ultra_comparison(rows):
    """The A2 ULTRA cells of the same graphs: medians over the three checkpoints."""
    by = {}
    for r in rows:
        if r.get("kind") == "eval" and r["ruleset"] == RULESET and r["graph"] in GRAPHS:
            by.setdefault(r["graph"], {}).setdefault(r["subset"], {})[r["ckpt"]] = _entry(r)
    return {g: {s: dict(_summary(list(by[g][s].values())), checkpoints=sorted(by[g][s]))
                for s in ("pooled", "affected")} for g in by}


def analyze_rows(rows, completed, ultra_rows=()):
    """Every complete seed is reported; W1F counts the seeds whose unit completed strictly
    before the cutoff and is judged only when all five of both graphs did."""
    by = {}
    for r in rows:
        if r.get("kind") == "eval" and r.get("mode") == "production":
            by.setdefault(r["graph"], {}).setdefault(int(r["seed"]), {})[r["subset"]] = r
    graphs = {}
    for gkey in GRAPHS:
        seeds = sorted(s for s, sub in by.get(gkey, {}).items()
                       if set(sub) == set(SUBSETS) and (gkey, s) in completed)
        on_time = [s for s in seeds if completed[(gkey, s)] < CUTOFF_UTC]
        g = {"seeds_complete": seeds, "seeds_on_time": on_time, "on_time_complete":
             on_time == list(SEEDS)}
        per = {s: {name: _entry(r) for name, r in by[gkey][s].items()} for s in seeds}
        g["per_seed"] = {str(s): v for s, v in per.items()}
        if seeds:
            med = {name: _summary([per[s][name] for s in seeds]) for name in SUBSETS}
            unaff = []
            for s in seeds:
                p, a = per[s]["pooled"], per[s]["affected"]
                o = (p["original"] * p["n_original"] - a["original"] * a["n_original"]) / \
                    (p["n_original"] - a["n_original"])
                mg = (p["merged"] * p["n_merged"] - a["merged"] * a["n_merged"]) / \
                    (p["n_merged"] - a["n_merged"])
                unaff.append(mg - o)
            med["unaffected_derived"] = {"delta": statistics.median(unaff),
                                         "abs_delta": statistics.median(abs(x) for x in unaff)}
            g["medians"] = med
            judged = [per[s] for s in on_time]
            if judged:
                aff = statistics.median(abs(v["affected"]["delta"]) for v in judged)
                pool = statistics.median(abs(v["pooled"]["delta"]) for v in judged)
                g["local_condition"] = bool(aff >= LOCAL_MIN and aff >= LOCAL_RATIO * pool)
        graphs[gkey] = g
    if all(graphs[g]["on_time_complete"] for g in GRAPHS):
        verdict = "supported" if all(graphs[g]["local_condition"] for g in GRAPHS) else "not supported"
    else:
        verdict = "incomplete"
    return {"W1F": verdict, "cutoff_utc": CUTOFF_UTC, "graphs": graphs,
            "ultra_a2": ultra_comparison(ultra_rows)}


def analyze():
    rows = read_jsonl(RESULTS)
    out = analyze_rows(rows, completion_times(rows), read_jsonl(A2_ULTRA_ROWS))
    with open(os.path.join(RUN_DIR, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"W1F: {out['W1F']}")
    for gkey, g in out["graphs"].items():
        if "medians" not in g:
            print(f"{gkey}: no complete seed")
            continue
        m, u = g["medians"], out["ultra_a2"].get(gkey, {})
        print(f"{gkey}: seeds {g['seeds_complete']} (on time {g['seeds_on_time']}); FLOCK pooled "
              f"{m['pooled']['original']:.4f} -> {m['pooled']['merged']:.4f} "
              f"(|delta| {m['pooled']['abs_delta']:.4f}); affected {m['affected']['original']:.4f} -> "
              f"{m['affected']['merged']:.4f} (|delta| {m['affected']['abs_delta']:.4f}, relative "
              f"{m['affected']['relative']:+.3f}, lower in {m['affected']['lower']} of "
              f"{len(g['seeds_complete'])} seeds); local condition "
              f"{ {True: 'holds', False: 'fails', None: 'not judged'}[g.get('local_condition')] }")
        if u:
            print(f"{gkey}: ULTRA checkpoint medians: pooled |delta| {u['pooled']['abs_delta']:.4f}, "
                  f"affected relative {u['affected']['relative']:+.3f}")


# ------------------------------------------------------------- selftest ----

def selftest():
    labels = np.array(["affected", "shifted", "affected", "identical", "affected"])
    tiers = np.array(["high", "", "low", "", "mid"])
    m = subset_masks(labels, tiers, np.array([0, 2, 3, 4]))
    assert m["pooled"].tolist() == [True] * 4
    assert m["affected"].tolist() == [True, True, False, True]
    assert (np.stack([m["affected_" + t] for t in TIERS]).sum(0) == m["affected"]).all()

    sizes = {"pooled": 100, "affected": 10, "affected_high": 3, "affected_mid": 3, "affected_low": 4}

    def rows_for(gkey, seed, pooled, affected):
        out = []
        for s in SUBSETS:
            o, mrg = (pooled if s == "pooled" else affected)
            out.append({"kind": "eval", "mode": "production", "graph": gkey, "seed": seed,
                        "subset": s, "mrr_original": o, "mrr_merged": mrg,
                        "n_queries_original": sizes[s], "n_queries_merged": sizes[s]})
        return out

    on_time = {(g, s): "2026-10-21T00:00:00+00:00" for g in GRAPHS for s in SEEDS}
    rows = [r for g in GRAPHS for s in SEEDS for r in rows_for(g, s, (0.20, 0.198), (0.15, 0.10))]
    res = analyze_rows(rows, on_time)
    assert res["W1F"] == "supported" and all(res["graphs"][g]["local_condition"] for g in GRAPHS)
    unaff = res["graphs"]["wk-50"]["medians"]["unaffected_derived"]["delta"]
    assert abs(unaff - ((0.198 * 100 - 0.10 * 10) / 90 - (0.20 * 100 - 0.15 * 10) / 90)) < 1e-12
    small = [r for g in GRAPHS for s in SEEDS for r in rows_for(g, s, (0.20, 0.19), (0.15, 0.13))]
    assert analyze_rows(small, on_time)["W1F"] == "not supported"
    partial = [r for r in rows if not (r["graph"] == "wk-100" and r["seed"] == 4)]
    assert analyze_rows(partial, on_time)["W1F"] == "incomplete"
    late = dict(on_time)
    late[("wk-50", 0)] = CUTOFF_UTC
    res_late = analyze_rows(rows, late)
    assert res_late["W1F"] == "incomplete"
    assert res_late["graphs"]["wk-50"]["seeds_complete"] == list(SEEDS)
    assert res_late["graphs"]["wk-50"]["seeds_on_time"] == [1, 2, 3, 4]
    empty = [dict(r, mrr_original=None, mrr_merged=None, n_queries_original=0, n_queries_merged=0)
             if r["subset"] == "affected_low" else r for r in rows]
    low = analyze_rows(empty, on_time)["graphs"]["wk-50"]["medians"]["affected_low"]
    assert low["n_original"] == 0 and low["delta"] is None
    assert parse_list("0,4", SEEDS, int) == [0, 4]
    try:
        parse_list("5", SEEDS, int)
    except SystemExit:
        pass
    else:
        raise AssertionError("seed 5 was accepted")
    print("selftest OK: subset masks partition the affected tiers; the local condition, W1F, "
          "incompleteness and the strict cutoff behave as specified; late seeds stay reportable; "
          "an empty tier is reported with its size; out-of-range seeds are refused")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--verify-commit", action="store_true")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--graphs", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    graphs = parse_list(args.graphs, GRAPHS, str)
    seeds = parse_list(args.seeds, SEEDS, int)
    if args.selftest:
        return selftest()
    if args.export:
        return export()
    if args.certify:
        return certify_all()
    if args.check:
        return check()
    if args.analyze:
        return analyze()
    if args.verify_commit:
        return verify_commit(args.commit)
    if args.smoke:
        return run("smoke", args.commit, graphs[:1], seeds[:1], resume=False)
    return run("production", args.commit, graphs, seeds, args.resume)


if __name__ == "__main__":
    main()
