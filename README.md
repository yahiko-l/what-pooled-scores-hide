# What Pooled Scores Hide: code and data

This bundle accompanies the WWW 2027 submission "What Pooled Scores Hide: Relation-Local
Costs of Web Schema Changes for Knowledge Graph Foundation Models". It holds the
experiment code and the results of the experiments the paper reports: the exact relabeling
controls with the consistent renaming, the entity-data perturbation and the interface factorial,
the displacement-matched contrast, and the three WWW 2027 amendments. Rerunning the experiments needs a CUDA GPU and the
public models and datasets listed below. The interface factorial is included as its collected
results only, without its runner or per-evaluation values.

## Contents

### `code/`

The runners of the reported experiments and the local modules they use (Python 3.10).

| Role | Files |
|---|---|
| WWW 2027 amendments | `www_strata_replication.py` (stratum-logged replication of the matched contrast), `www_construction.py` (construction without edgeless relations), `www_schema_merge.py` (documented predicate merges), `www_gpu.py` (GPU admission on a shared server), `www_analysis.py` (frozen endpoints and decision rules), `www_flock_merge.py` (FLOCK under the predicate merges of WK-50 and WK-100, with its own checks and analysis) |
| Predicate-merge specifications | `MERGE_SPEC.json`, `MERGE_SPEC_A2.json` (frozen merge specifications of the first and second amendments: groups, sources with revisions and sha256, checks); `make_merge_spec.py`, `make_merge_spec_a2.py` (regenerate them byte for byte from the pinned sources; the second imports the first from the same directory) |
| Displacement-matched contrast | `ultra_matched_ladder.py`, `flock_matched_ladder.py` (runners); `matched_pairs.py`, `matched_construction.py`, `matched_feasibility.py` (construction, certificates, invocation gates, judgeability) |
| Exact relabeling controls | `r1_ladder.py` (ULTRA), `trix_ladder.py`, `flock_ladder.py`; `joint_rename_control.py` (consistent renaming of the whole input on ULTRA); `native_perturbation.py` (entity-data perturbation, which also supplies `split_halves` to the merge runner) |
| Harnesses and shared code | `pilot_harness.py` (ULTRA), `trix_harness.py`, `flock_harness.py`, `tunit_common.py` |

The ULTRA, TRIX and FLOCK source trees are not included; they are public (see
[Public code, checkpoints and data](#public-code-checkpoints-and-data)) and are cloned next to
these modules.

### `results/`

`results/<RUN>/` keeps the run directory's original name. For the exact controls, the
consistent renaming, the entity-data perturbation and the interface factorial, the files hold
the graphs, relabelings and values the paper reports. The rows of the matched contrast and of the three WWW 2027
amendments are complete, as the frozen analysis below reads them whole.

| Run directory | Files | In the paper |
|---|---|---|
| `R1`, `R1_censusfix`, `R1_OOF_3g`, `R1_XCKPT_4g`, `R1_XCKPT_50g` | `r1_ladder_results.jsonl` (`R1_censusfix`: `metrics.json`; `R1_OOF_3g` also `metrics.json`) | ULTRA exact controls: per-seed MRR and recovery of the role-preserving (`L4`) and unrestricted (`L4loose`) relabelings with the unperturbed baseline (`L0`), checkpoint 3g from `R1` (FB graphs) and `R1_OOF_3g` (InGram graphs), 4g and 50g from `R1_XCKPT_*`. `R1` also holds the endpoint randomization (`L5`) on the FB graphs, `R1_censusfix` the degree-preserving rewiring (`L1`) on the FB graphs, and `R1_OOF_3g` the fraction of relations a role-preserving relabeling can move |
| `TRIX_CLAIM_prereg_v2` | `trix_ladder_results.jsonl`, `trix_summary.json` | TRIX exact controls (`L0`, `L4`, `L4loose`) and the fraction of relations a role-preserving relabeling can move |
| `FLOCK_CLAIM_prereg_v2` | `flock_ladder_results.jsonl` | FLOCK exact controls: baseline (`L0`), role-preserving (`L4`) and unrestricted (`L4loose`) relabelings of the message graph's edge types, and the consistent control (`D-i`) |
| `JOINT_RENAME_CONTROL_prereg_20260903` | `rows.jsonl`, `summary.json` | Consistent renaming (FB v1 to v4, checkpoint 3g, five seeds): one permutation applied to the triples, queries, answers and filters, with the relation graph rebuilt by ULTRA's own constructor; per cell the moved base relations, MRR before and after, recovery and per-query rank agreement |
| `NATIVE_PERTURBATION_CROSS_prereg_20260903_A1` | `rows.jsonl`, `summary.json` | Entity-data perturbation (FB v1 to v4, checkpoint 3g): recovery and Jaccard similarity of the rebuilt relation graph per dose and seed, and the dose curve |
| `IFACE_prereg_v5` | `IFACE_RESULTS.json` | Interface factorial: recovery of each arm per cell, per graph and over the panel (24 cells); arm `Y01` restores the edge-label lookup, `Y10` the query lookup, `Y00` neither and `Y11` both |
| `MATCHED_FEASIBILITY` | `feasibility_r3.json` | Frozen judgeability and exposure statistics of the matched contrast, which left no TRIX graph judgeable |
| `MATCHED_ULTRA_prereg_v3` | `ultra_matched_results.jsonl` | Per-draw pooled and stratum rows of the matched contrast on ULTRA, also the archived side of the determinism check of experiment 1 |
| `MATCHED_FLOCK_prereg_v3` | `flock_matched_results.jsonl` | Per-draw rows of the matched contrast on FLOCK |
| `WWW_STRATA_ultra_legacy`, `WWW_STRATA_ultra_populated`, `WWW_STRATA_flock_populated` | `*_strata_results.jsonl` | Production rows of experiment 1 |
| `WWW_MERGE` | `www_merge_results.jsonl` | Production rows of experiment 2 |
| `WWW_STRATA_A2_ultra_populated` | `ultra_strata_results.jsonl` | Production rows of experiment 3 on ULTRA (seven held-out graphs) |
| `WWW_STRATA_A2_flock_populated` | `flock_strata_results.jsonl`, `ATTEMPTS.jsonl` | Production rows of experiment 3 on FLOCK (WK-50 and WK-100) and the attempt ledger whose success times `exp3` checks against the completion cutoff |
| `WWW_MERGE_A2` | `www_merge_results.jsonl` | Production rows of experiment 4 (merges on six held-out graphs) |
| `WWW_MERGE_A3_FLOCK` | `flock_merge_results.jsonl` | Production rows of the third amendment (FLOCK under the merges of WK-50 and WK-100, five seeds) |
| `WWW_FLOCK_GRAIL_BASELINE` | per-graph rows, report | Reproduction of FLOCK's published zero-shot MRR on FB v1 to v4 with the loader used for these graphs, quoted in the setting section |

Each entry of the exact-control table is the median over five seeds of MRR(relabeling, seed) /
MRR(unperturbed graph); for FLOCK the unperturbed MRR is evaluated at the same seed.

## Paths in comments and result files

Comments and result fields refer to the layout of the original repository:

| Original path | In this bundle |
|---|---|
| `idea-stage/pilot/<module>.py` | `code/<module>.py` |
| `idea-stage/pilot/runs/<RUN>/` | `results/<RUN>/` |
| `refine-logs/www2027/MERGE_SPEC.json`, `refine-logs/www2027/make_merge_spec.py`, and their `_A2` counterparts | `code/` |
| `idea-stage/pilot/{ULTRA,TRIX,FLOCK}/` | public repositories, cloned into `code/` |

The `spec` and `ckpt` fields of the result files hold these repository-relative paths.

## Environment

The decision rules below run on a CPU with Python 3.10 or later and the standard library;
`www_flock_merge.py --analyze` also needs numpy and PyTorch, and the two merge-specification
generators need numpy. Checked with Python 3.12.7 and numpy 1.26.4.

Experiments (GPU) ran with Python 3.10, PyTorch 2.1.2 (CUDA 12.1 build), PyTorch Geometric 2.4.0 and
numpy 1.26.4. They also need torch-scatter, ninja, easydict, pyyaml, jinja2, scipy and
scikit-learn. ULTRA and TRIX compile their `rspmm` kernel on first use, which needs a CUDA toolkit
with `nvcc` and `CUDA_HOME` set. FLOCK additionally needs pybind11 and its random-walk extension
(`bash install_walker.sh` in the FLOCK repository). The WWW runners call `nvidia-smi` for GPU
admission.

## Public code, checkpoints and data

| Model | Repository | Commit | Checkpoints used (shipped in the repository) |
|---|---|---|---|
| ULTRA | https://github.com/DeepGraphLearning/ULTRA | `427966ad8ed60420eef034063d44f3153addff90` | `ckpts/ultra_3g.pth`, `ckpts/ultra_4g.pth`, `ckpts/ultra_50g.pth` |
| TRIX | https://github.com/yuchengz99/TRIX | `7596e14eefefe89e61396205a0550172cadeddb0` | `entity_prediction.pth` |
| FLOCK | https://github.com/jw9730/flock | `f35103d25a78bdf4075de5c673a51de4979aa4d7` | `checkpoints/flock_entity.pth`, config `src_entity/config/zeroshot_inductive/n16_ensemble16.yaml` |

Datasets, downloaded on first use by each model's own dataset classes:

- FB15k-237 inductive splits v1 to v4 (GraIL, https://github.com/kkteru/grail). ULTRA's
  `FB15k237Inductive` stores them under `code/kg-datasets/`; TRIX under
  `code/TRIX/trix-kg-datasets/`; FLOCK under the root set in its config.
- InGram NL-25, NL-50, NL-75, NL-100 and WK-75 (https://github.com/bdi-lab/InGram), loaded through
  ULTRA's `NLIngram` and `WKIngram` classes (`tunit_common.load_ingram`), which download the raw
  files, and through the corresponding classes of TRIX and FLOCK.
- InGram FB-25, FB-50, FB-75, FB-100, WK-50, WK-100 and NL-0, the held-out graphs of the second
  WWW amendment, loaded on ULTRA through `FBIngram`, `WKIngram` and `NLIngram` with the filter of
  `tunit_common.load_ingram` (`www_strata_replication.a2_dataset`, `www_schema_merge.load_ingram_a2`),
  and on FLOCK through `flock_ladder.build_graph`.

Sources of the predicate merges, each pinned in `code/MERGE_SPEC.json` (`sources`,
`reference_not_run` and `data`) with its URL, revision or commit, and sha256:

| Source | Pinned at | Path expected by `make_merge_spec.py` |
|---|---|---|
| Wikidata, WikiProject Freebase/Mapping | revision 254788053 (2015-10-04) | `dl_wd_fbmapping_2015/mapping_rev254788053.wikitext` |
| google/freebase-wikidata-converter, `convert/lib.php` | commit `4e18711137030c41ee4ea558e2b7eb8292211b8d` | `dl_gh_converter_src/convert_lib.php` |
| YAGO 4.5 shapes, `input-data/00-shapes.ttl` (yago-naga/yago-4.6) | commit `4ef3c07ccf16675747427fe3e989a6ce96f19bae` | `dl_yago45_shapes_4ef3c07/00-shapes.ttl` |
| Previous version of the same file | commit `9446700f62931a445b35237b914e83f19a48b507` | `dl_yago45_shapes_9446700f/00-shapes.ttl` |
| YAGO 4.6 development schema (reference only, not run) | commit `5f84fa960fca294ad3784c974a78a127b21cd358` | `dl_yago46_schema/yago-schema.ttl` |
| GraIL `data/fb237_v{1,2,3,4}_ind/{train,valid,test}.txt` | commit `2a3dffa719518e7e6250e355a2fb37cd932de91e` | `dl_grail/fb237_{v}_ind/{split}.txt` |
| InGram `data/WK-75/{msg,valid,test}.txt` | commit `f2554dd0c684bad71d3e01d534f2f7814cf28644` | `dl_ingram_wk75/{split}.txt` |

```bash
python3 -I code/make_merge_spec.py <download_dir> regenerated_MERGE_SPEC.json
cmp regenerated_MERGE_SPEC.json code/MERGE_SPEC.json
```

`code/MERGE_SPEC_A2.json` uses the same mapping and YAGO sources and the InGram raw files of the
held-out graphs, pinned in its `data` section (commit `f2554dd0c684bad71d3e01d534f2f7814cf28644`,
expected under `dl_ingram_a2/{graph}/{split}.txt`):

```bash
python3 -I code/make_merge_spec_a2.py <download_dir> regenerated_MERGE_SPEC_A2.json
cmp regenerated_MERGE_SPEC_A2.json code/MERGE_SPEC_A2.json
```

## Decision rules from the results (CPU)

The decision rules of the WWW amendments, from the frozen analysis module:

```bash
python3 code/www_analysis.py exp1 \
    results/WWW_STRATA_ultra_legacy/ultra_strata_results.jsonl \
    results/WWW_STRATA_ultra_populated/ultra_strata_results.jsonl \
    results/WWW_STRATA_flock_populated/flock_strata_results.jsonl \
    --archived results/MATCHED_ULTRA_prereg_v3/ultra_matched_results.jsonl --out exp1.json
python3 code/www_analysis.py exp2 results/WWW_MERGE/www_merge_results.jsonl --out exp2.json
python3 code/www_analysis.py exp3 \
    results/WWW_STRATA_A2_ultra_populated/ultra_strata_results.jsonl \
    results/WWW_STRATA_A2_flock_populated/flock_strata_results.jsonl \
    --attempts results/WWW_STRATA_A2_flock_populated/ATTEMPTS.jsonl --out exp3.json
python3 code/www_analysis.py exp4 results/WWW_MERGE_A2/www_merge_results.jsonl --out exp4.json
```

`exp1` prints the determinism check (1,932 required values, maximum difference 0) and the verdicts
P1 and P2 (both hold) and NL-25 (no cell replicates); `exp2` prints M1 (not supported) and M2
(supported, 10 of 12 qualifying cells); `exp3` prints Q1 (fails: WK-50 replicates on one
checkpoint), Q2 (holds: FB-25, FB-50 and FB-75 pass), Q3 (fails: FLOCK replicates on WK-50 and
not on WK-100) and the NL-0 cells (none replicates);
`exp4` prints M1′ (not supported: only FB-25 meets the local condition), W1 (supported on WK-50 and
WK-100) and M2′ (supported, 7 of 9 qualifying cells).

The third amendment's rule comes from its frozen runner, which imports numpy and PyTorch but needs no
GPU in this mode and reads its rows from `code/runs/`:

```bash
mkdir -p code/runs
cp -r results/WWW_MERGE_A3_FLOCK results/WWW_MERGE_A2 code/runs/
(cd code && python3 www_flock_merge.py --analyze)
```

It prints W1F (supported: the local condition holds on WK-50 and WK-100) and writes the per-seed
values and their medians to `code/runs/WWW_MERGE_A3_FLOCK/analysis.json`.

## Rerun the experiments (GPU)

Set up the backbones inside `code/`:

```bash
cd code
git clone https://github.com/DeepGraphLearning/ULTRA.git ULTRA
git -C ULTRA checkout 427966ad8ed60420eef034063d44f3153addff90
git clone https://github.com/yuchengz99/TRIX.git TRIX
git -C TRIX checkout 7596e14eefefe89e61396205a0550172cadeddb0
cp TRIX/config/run_entity_inductive.yaml TRIX/config/run_entity_inductive_local.yaml
git clone https://github.com/jw9730/flock.git ~/flock     # FLOCK's configs expect ~/flock
git -C ~/flock checkout f35103d25a78bdf4075de5c673a51de4979aa4d7
(cd ~/flock && bash install_walker.sh)
ln -s ~/flock FLOCK                                         # or export FLOCK_REPO=~/flock
export CUDA_HOME=/usr/local/cuda                            # your CUDA toolkit
```

`trix_harness.py` reads only the model section of `run_entity_inductive_local.yaml`, which is the
same as in the repository's `run_entity_inductive.yaml`. All runs write to `code/runs/<RUN>/`; set
`CUDA_VISIBLE_DEVICES` to one GPU. The matched-contrast and WWW runners refuse to overwrite an
existing results file, and `--resume` continues an interrupted run.

Exact relabeling controls and the entity-data perturbation:

```bash
python r1_ladder.py --ladder L4,L4loose,L5 --seeds 5 --fb-versions v1,v2,v3,v4 --wn-null 0 \
    --explore '' --run-id R1
python r1_ladder.py --ladder L1 --seeds 5 --fb-versions v1,v2,v3,v4 --wn-null 0 --explore '' \
    --anneal-steps 2000000 --run-id R1_censusfix
python r1_ladder.py --ladder L1,L4,L4loose,L5 --seeds 5 --fb-versions '' --wn-null 0 --explore '' \
    --claim-graphs nl-50,nl-75,nl-100,wk-75 --ckpt ultra_3g --run-id R1_OOF_3g
python r1_ladder.py --ladder L1,L4,L4loose,L5 --seeds 5 --fb-versions v1,v2,v3,v4 --wn-null 0 --explore '' \
    --claim-graphs nl-50,nl-75,nl-100,wk-75 --ckpt ultra_4g --run-id R1_XCKPT_4g
python r1_ladder.py --ladder L1,L4,L4loose,L5 --seeds 5 --fb-versions v1,v2,v3,v4 --wn-null 0 --explore '' \
    --claim-graphs nl-50,nl-75,nl-100,wk-75 --ckpt ultra_50g --run-id R1_XCKPT_50g
python trix_ladder.py --fb-versions v1,v2,v3,v4 --claim-graphs nl-50,nl-75,nl-100,wk-75 --seeds 5 \
    --run-id TRIX_CLAIM_prereg_v2
python flock_ladder.py --claim-graphs nl-50,nl-75,nl-100,wk-75 --explore-graphs "" \
    --arms L4,L4loose,D-i --seeds 5 --run-id FLOCK_CLAIM_prereg_v2
python joint_rename_control.py                 # writes runs/JOINT_RENAME_CONTROL_prereg_20260903
python native_perturbation.py --mode cross     # writes runs/NATIVE_PERTURBATION_CROSS_prereg_20260903_A1
```

Displacement-matched contrast. The frozen invocation gate accepts only the full production panel,
which is the default:

```bash
python ultra_matched_ladder.py
python flock_matched_ladder.py
```

First WWW amendment. Production requires `--commit`, a 40-character commit of the code being run;
the archived rows carry the frozen commit `726ade72e7491e74695e6fd0ea477436e241a53f`.

```bash
COMMIT=726ade72e7491e74695e6fd0ea477436e241a53f
python www_strata_replication.py --backbone ultra --construction legacy --commit $COMMIT
python www_strata_replication.py --backbone ultra --construction populated --commit $COMMIT
python www_strata_replication.py --backbone flock --construction populated --commit $COMMIT
python www_schema_merge.py --spec MERGE_SPEC.json --commit $COMMIT
python www_analysis.py exp1 runs/WWW_STRATA_ultra_legacy/ultra_strata_results.jsonl \
    runs/WWW_STRATA_ultra_populated/ultra_strata_results.jsonl \
    runs/WWW_STRATA_flock_populated/flock_strata_results.jsonl \
    --archived ../results/MATCHED_ULTRA_prereg_v3/ultra_matched_results.jsonl --out exp1.json
python www_analysis.py exp2 runs/WWW_MERGE/www_merge_results.jsonl --out exp2.json
```

Second WWW amendment, run with the revision in `code/` and its frozen commit:

```bash
COMMIT=7f8020fb6bc792417933a007a3e7995411a512e8
python www_strata_replication.py --amendment a2 --backbone ultra --construction populated --commit $COMMIT
python www_strata_replication.py --amendment a2 --backbone flock --construction populated --commit $COMMIT
python www_schema_merge.py --amendment A2 --spec MERGE_SPEC_A2.json --commit $COMMIT
python www_analysis.py exp3 runs/WWW_STRATA_A2_ultra_populated/ultra_strata_results.jsonl \
    runs/WWW_STRATA_A2_flock_populated/flock_strata_results.jsonl \
    --attempts runs/WWW_STRATA_A2_flock_populated/ATTEMPTS.jsonl --out exp3.json
python www_analysis.py exp4 runs/WWW_MERGE_A2/www_merge_results.jsonl --out exp4.json
```

Third WWW amendment, with `www_flock_merge.py` as frozen and its commit. `--export` builds the
merged graphs with `www_schema_merge.py` and certifies them against the structure rows in
`runs/WWW_MERGE_A2/` (copy them there from `results/`); it reads the specification at
`refine-logs/www2027/MERGE_SPEC_A2.json` under the root that `matched_pairs.REPO_ROOT` names, two
levels above `code/`, so copy `code/MERGE_SPEC_A2.json` there:

```bash
COMMIT=61e1512c4a5de04ad8490b60a0994ba147825b05
python www_flock_merge.py --export        # CPU, ULTRA's dataset classes
python www_flock_merge.py --certify       # CPU, FLOCK's dataset classes
python www_flock_merge.py --commit $COMMIT
python www_flock_merge.py --analyze
```

FLOCK production can be split across processes by disjoint `--graphs` and `--draws` (the third
amendment's runner by `--graphs` and `--seeds`), each later process with `--resume`. FLOCK runs
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which bounds its reserved memory and does
not change its computation.

## Notes on this copy

- `code/` is the revision frozen by `prereg-www2027-a2`, with `www_flock_merge.py` added from the
  revision frozen by `prereg-www2027-a3`. That file is the frozen bytes, so its SHA-256 equals the
  digest each production unit of the third amendment recorded; the other modules did not change
  between the two tags. The production rows of the first WWW
  amendment were written by the revision frozen by `prereg-www2027`, which differs only in four
  `www_*` modules: the later revision adds the second amendment's graphs, rules and checks and
  raises the GPU sharing threshold, and its `exp1` and `exp2` reproduce the first amendment's
  results from those rows exactly. Some archived results were written by earlier revisions.
  The `MATCHED_*_prereg_v3` results
  predate the move of the arm construction from `matched_pairs.py` into `matched_construction.py`;
  the WWW determinism check reran the twelve archived ULTRA cells on the FB graphs with the revision
  frozen by `prereg-www2027` and reproduced all 2,184 pooled and stratum MRRs it compared, with
  maximum difference 0.
  `R1_OOF_3g`, `R1_XCKPT_4g` and `R1_XCKPT_50g` were written by this revision of `r1_ladder.py`;
  `R1` and `R1_censusfix` by an earlier one, before amendment 1 added the checkpoint option and the
  claim-graph mode. `JOINT_RENAME_CONTROL_prereg_20260903` was written by this revision of
  `joint_rename_control.py`. `R1_censusfix` already used this revision's reciprocal-pair-preserving `L1`
  construction.
- The interface factorial's runner is not part of `code/`. Its per-evaluation values are stored
  only in that run's sealed value file (145 MB), which is not included; `IFACE_RESULTS.json` is the
  collected result the paper's values are read from.
- `pilot_harness.py` and `tunit_common.py` keep only the functions that the reported runs call.
- This copy is anonymized: identifying and internal references were removed or replaced with
  neutral wording or `[redacted]`. In the code this touched comments, docstrings, output strings
  and the default FLOCK checkout path, which became relative to `code/`. In the result files,
  absolute paths in `spec` and `ckpt` fields became repository-relative. No design, rule,
  threshold or number was changed.
