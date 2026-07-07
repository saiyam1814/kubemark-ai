# Published Benchmark Reports

Curated, machine-readable comparison reports committed to the repo so results
are citable and diffable. Raw run artifacts (full pod logs, per-repeat JSONs)
stay in the gitignored `results/` directory — they can contain
cluster/node/project identifiers and are not published.

Each file is the `comparison.json` emitted by `scripts/compare-runtimes.sh`:
per-runtime metrics plus `overhead_vs_baseline` deltas, tagged with
`"vcluster": true` when the run executed inside ephemeral vClusters.

| Report | Hardware | Runtimes | Benchmarks | Notes |
|---|---|---|---|---|
| `compare-20260707-083442.json` | GKE 1.35, 1× L4 | runc, gVisor | coldstart | Warmup run (node scale-up + image pull inflate gVisor sandbox startup) |
| `compare-20260707-084508.json` | GKE 1.35, 1× L4 | runc, gVisor | NCCL, coldstart | |
| `compare-20260707-140126.json` | GKE 1.35, 1× L4 | runc, gVisor, vNode | NCCL, coldstart | Headline host-cluster numbers |
| `compare-20260707-143651.json` | GKE 1.35, 1× L4 | runc, gVisor, vNode | coldstart | Inside ephemeral vClusters |

Human-readable analysis: [runtime-comparison-summary.md](../runtime-comparison-summary.md)

## Publishing checklist for new reports

Before committing a new report here:

1. Copy `results/compare-<id>/comparison.json` to this directory.
2. Verify it contains **no** cluster names, node names, project IDs, account
   emails, URLs, or credentials (`grep` it — the standard report schema has
   none, but check anything hand-edited).
3. Add a row to the table above with hardware and caveats.
4. Update `docs/runtime-comparison-summary.md` if the result changes the
   conclusions.
