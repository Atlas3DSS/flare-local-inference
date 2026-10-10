"""Summarize TP graph envelopes from GLM HTTP benchmark Nsight traces.

Flare graph names come from its prepared catalog. Peer launches are matched by
the ordered native dispatch sequence, with equal launch counts required.
vLLM activities are correlated with worker forward/sample NVTX ranges and
filtered using actual decode metadata. Durations are wall time on the GPU;
concurrent rank durations are never added together.
"""

import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

from qualify.gpu_replay_analysis import union_ns


def distribution(values):
    values = sorted(values)
    return {"count": len(values), "mean_ms": statistics.mean(values) / 1e6,
            "median_ms": statistics.median(values) / 1e6,
            "p95_ms": values[int((len(values) - 1) * .95)] / 1e6}


def pid(global_id):
    return (global_id >> 24) & 0xffffff


def read_trace(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    names = ("StringIds", "NVTX_EVENTS", *("CUPTI_ACTIVITY_KIND_" + suffix for suffix in
             ("GRAPH_TRACE", "KERNEL", "MEMCPY", "MEMSET", "RUNTIME", "DRIVER")))
    data = {name: [dict(row) for row in db.execute("select * from " + name)] if name in tables else []
            for name in names}
    db.close()
    return data


def flare(data, directory):
    catalog = json.loads((directory / "graphs.json").read_text())
    names = {graph["id"]: graph["name"] for graph in catalog["graphs"]}
    ranks = defaultdict(list)
    for graph in data["CUPTI_ACTIVITY_KIND_GRAPH_TRACE"]:
        ranks[pid(graph["globalPid"])].append(graph)
    for graphs in ranks.values():
        graphs.sort(key=lambda graph: graph["start"])
    root = ranks[catalog["pid"]]
    if not root or any(len(graphs) != len(root) for graphs in ranks.values()):
        raise ValueError("rank graph launch counts differ")
    indices = [i for i, graph in enumerate(root) if names[graph["graphExecId"]] == "decode1"]
    if len(indices) != 765:
        raise ValueError(f"expected 765 batch-one decode graphs; found {len(indices)}")
    report = {"ranks": {process: distribution([graphs[i]["end"] - graphs[i]["start"] for i in indices])
                        for process, graphs in ranks.items()}, "root_pid": catalog["pid"]}
    envelopes = []
    for i in indices:
        graphs = [rank[i] for rank in ranks.values()]
        if max(g["start"] for g in graphs) >= min(g["end"] for g in graphs):
            raise ValueError("corresponding rank decode graphs do not overlap")
        envelopes.append(max(g["end"] for g in graphs) - min(g["start"] for g in graphs))
    report["all_rank_envelope"] = distribution(envelopes)
    report["repeat_mean_ms"] = [statistics.mean(envelopes[i:i + 255]) / 1e6 for i in range(0, 765, 255)]
    return report


def vllm(data, steps_path):
    strings = {row["id"]: row["value"] for row in data["StringIds"]}
    scopes = defaultdict(list)
    for row in data["NVTX_EVENTS"]:
        name = row.get("text") or strings.get(row.get("textId"), "")
        if name.startswith(("vllm.forward.", "vllm.sample.")) and row["end"] is not None:
            scopes[row["globalTid"]].append(dict(row, name=name))
    for rows in scopes.values():
        rows.sort(key=lambda row: row["start"])
    starts = {tid: [row["start"] for row in rows] for tid, rows in scopes.items()}
    correlations = {}
    for table in ("RUNTIME", "DRIVER"):
        for api in data["CUPTI_ACTIVITY_KIND_" + table]:
            tid = api["globalTid"]
            if tid not in scopes:
                continue
            index = bisect.bisect_right(starts[tid], api["start"]) - 1
            if index >= 0 and api["start"] < scopes[tid][index]["end"]:
                correlations[(pid(tid), api["correlationId"])] = scopes[tid][index]["name"]
    activities = defaultdict(lambda: defaultdict(list))
    for kind in ("GRAPH_TRACE", "KERNEL", "MEMCPY", "MEMSET"):
        for gpu in data["CUPTI_ACTIVITY_KIND_" + kind]:
            process = pid(gpu["globalPid"])
            scope = correlations.get((process, gpu["correlationId"]))
            if scope:
                activities[(process, scope)][kind].append(gpu)
    processes = sorted({process for process, scope in activities})
    steps = json.loads(steps_path.read_text())
    decode = [step for step in steps if not step["has_prefill"]]
    if len(decode) != 765:
        raise ValueError(f"expected 765 batch-one decode steps; found {len(decode)}")
    reports, envelopes = {}, []
    for process in processes:
        graph_times, busy_times, envelope_times = [], [], []
        for step in decode:
            if (step["batch"] != 1 or step["tokens"] != 1 or step["padded_tokens"] != 1
                    or not 512 <= step["positions"][0] <= 766):
                raise ValueError(f"unexpected decode geometry: {step}")
            groups = [activities[(process, f"vllm.{scope}.{step['index']}")] for scope in ("forward", "sample")]
            graphs = [row for group in groups for row in group["GRAPH_TRACE"]]
            all_gpu = [row for group in groups for rows in group.values() for row in rows]
            if not graphs:
                raise ValueError(f"no model graph for process {process}, step {step['index']}")
            graph_times.append(union_ns(graphs))
            busy_times.append(union_ns(all_gpu))
            envelope_times.append(max(row["end"] for row in all_gpu) - min(row["start"] for row in all_gpu))
        reports[process] = {"graphs": distribution(graph_times), "forward_sample_gpu_busy": distribution(busy_times),
                            "forward_sample_envelope": distribution(envelope_times)}
    for step in decode:
        rows = [row for process in processes for scope in ("forward", "sample")
                for group in activities[(process, f"vllm.{scope}.{step['index']}")].values() for row in group]
        envelopes.append(max(row["end"] for row in rows) - min(row["start"] for row in rows))
    return {"ranks": reports, "all_rank_envelope": distribution(envelopes),
            "repeat_mean_ms": [statistics.mean(envelopes[i:i + 255]) / 1e6 for i in range(0, 765, 255)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("flare", "vllm"))
    parser.add_argument("trace", type=Path)
    parser.add_argument("directory", type=Path, help="HTTP results and Flare graph catalog")
    parser.add_argument("--steps", type=Path, help="vLLM rank-zero step metadata")
    args = parser.parse_args()
    results = json.loads((args.directory / "results.json").read_text())
    if [row["phase"] for row in results] != ["warmup", *["unprofiled"] * 3, *["profiled"] * 3]:
        raise ValueError("HTTP measurements are incomplete")
    data = read_trace(args.trace)
    report = flare(data, args.directory) if args.engine == "flare" else vllm(data, args.steps)
    report["http_tokens_per_second"] = [row["delivery"]["tokens_per_second"] for row in results
                                        if row["phase"] == "unprofiled"]
    (args.directory / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
