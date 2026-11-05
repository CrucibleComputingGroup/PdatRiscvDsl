#!/usr/bin/env python3
"""Compare legacy SVA against an immutable Git revision, byte for byte.

This is a generation-only audit: it never synthesizes or modifies a processor.
All discovered DSL inputs, including baseline failures, are recorded. The CLI
path covers primary and timing sidecar SVA; the hierarchical API path covers
operand assumptions emitted separately for submodules. A fixed hash seed makes
set iteration reproducible on both sides, rather than normalizing the output.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

BASELINE = "3fe0d81f0d8a2fed326fbd1d4095b1bd9f3b594a"
LEGACY_MODULES = (
    "__init__.py", "parser.py", "config.py", "codegen.py", "codegen_v2_aig.py",
    "dtype_codegen.py", "aig_constraints.py", "bdd_ops.py", "encodings.py",
    "hierarchical_synth.py", "rtl_patcher.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def discover(repo, other_repos):
    """Include tracked source DSLs plus present untracked source examples."""
    sources = {}
    for location in [repo, *other_repos]:
        if not (location / ".git").exists():
            continue
        tracked = set(git(location, "ls-files", "*.dsl").splitlines())
        candidates = set(tracked)
        for dirname in ("examples", "workload", "Motivation", "experiments/reg_count", "tests/regression/fixtures"):
            folder = location / dirname
            if folder.exists():
                candidates.update(str(p.relative_to(location)) for p in folder.rglob("*.dsl"))
        for rel in sorted(candidates):
            # New frontend/golden pairs have their own frozen-generator tests;
            # keep the historical experiment corpus a separate audit.
            if location == repo and rel.startswith("examples/frontend/"):
                continue
            path = location / rel
            if path.is_file():
                key = location.name + "/" + rel
                sources[key] = {"path": str(path.resolve()), "tracked": rel in tracked,
                                "sha256": sha(path.read_bytes()), "repository_head": git(location, "rev-parse", "HEAD")}
    return sources


def worker(package, requests_path, output):
    # A separate interpreter and cwd prevent imports from the current checkout
    # from contaminating the baseline run.
    import pdat_dsl
    from pdat_dsl.__main__ import main
    from pdat_dsl.codegen import generate_module_dtype_assertions
    from pdat_dsl.config import ModuleConfig, SignalConfig, load_config
    from pdat_dsl.parser import parse_dsl

    package = Path(package).resolve()
    assert Path(pdat_dsl.__file__).resolve().parent == package / "pdat_dsl"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(requests_path).read_text())
    results = {}
    for index, case in enumerate(cases):
        dest = output / str(index)
        dest.mkdir()
        record = {"input": case["input"], "mode": case["mode"], "config": case["config"], "artifacts": {}}
        diagnostic = io.StringIO()
        with contextlib.redirect_stdout(diagnostic), contextlib.redirect_stderr(diagnostic):
            try:
                if case["mode"] == "codegen-cli":
                    sys.argv = ["pdat-dsl", "codegen", case["source"], str(dest / "assumptions.sv")]
                    if case["config"] != "default":
                        sys.argv += ["--config", str(package / "configs" / (case["config"] + ".yaml"))]
                    record["command"] = list(sys.argv)
                    code = main()
                    if code not in (None, 0):
                        raise RuntimeError("CLI returned " + str(code))
                elif case["mode"] in ("hierarchical-dtype-api", "hierarchical-config-dtype-api"):
                    ast = parse_dsl(Path(case["source"]).read_text())
                    if case["mode"] == "hierarchical-dtype-api":
                        config = ModuleConfig(name="compat_ex", module_path="core.ex", signals=SignalConfig())
                        width = int(case["config"])
                    else:
                        core = load_config(None if case["config"] == "default" else package / "configs" / (case["config"] + ".yaml"))
                        config = ModuleConfig(name="compat_ex", module_path=core.injection.module_path, signals=core.signals)
                        width = core.data_width
                    code = generate_module_dtype_assertions(ast.rules, config, data_width=width)
                    (dest / "module_dtype.sv").write_text(code)
                else:
                    raise ValueError("Unsupported audit mode: " + case["mode"])
                record["status"] = "generated"
            except (Exception, SystemExit) as error:
                record["status"] = "error"
                record["error_type"] = type(error).__name__
                record["error"] = str(error)
        if record["status"] == "error":
            (dest / "diagnostic.txt").write_text(diagnostic.getvalue())
            record["diagnostic_file"] = str(dest / "diagnostic.txt")
        for file in sorted(dest.glob("*.sv")):
            data = file.read_bytes()
            record["artifacts"][file.name] = {"sha256": sha(data), "bytes": len(data), "file": str(file)}
        results[str(index)] = record
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def compare(before, after):
    """Require equal status and errors, and compare actual artifact bytes."""
    mismatches = []
    for key in sorted(set(before) | set(after), key=int):
        if key not in before or key not in after:
            mismatches.append({"case": key, "reason": "missing case"})
            continue
        left, right = before[key], after[key]
        if (left["status"], left.get("error_type"), left.get("error")) != (right["status"], right.get("error_type"), right.get("error")):
            mismatches.append({"case": key, "reason": "changed completion or error", "before": left, "after": right})
        if set(left["artifacts"]) != set(right["artifacts"]):
            mismatches.append({"case": key, "reason": "changed artifact set"})
            continue
        for name in left["artifacts"]:
            if Path(left["artifacts"][name]["file"]).read_bytes() != Path(right["artifacts"][name]["file"]).read_bytes():
                mismatches.append({"case": key, "reason": "SVA bytes changed", "artifact": name})
    return mismatches


def audit(repo, baseline_ref, output, other_repos, hash_seed="0", corpus_snapshot=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_head = git(repo, "rev-parse", baseline_ref)
    if corpus_snapshot:
        snapshot = json.loads(corpus_snapshot.read_text())["inputs"]
        inputs = {name: {key: value for key, value in item.items() if key != "source_text"}
                  for name, item in snapshot.items()}
    else:
        inputs = discover(repo, other_repos)
    requests = []
    inputs_dir = output / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for index, (name, item) in enumerate(inputs.items()):
        frozen = inputs_dir / (str(index) + ".dsl")
        data = snapshot[name]["source_text"].encode() if corpus_snapshot else Path(item["path"]).read_bytes()
        if sha(data) != item["sha256"]:
            raise ValueError("Input does not match recorded SHA-256: " + name)
        frozen.write_bytes(data)
        for config in ("default", "ibex", "boom", "rocket", "riscvsinglecycle"):
            requests.append({"input": name, "source": str(frozen), "mode": "codegen-cli", "config": config})
        for width in (32, 64):
            requests.append({"input": name, "source": str(frozen), "mode": "hierarchical-dtype-api", "config": str(width)})
    request_path = output / "requests.json"
    request_path.write_text(json.dumps(requests, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="crisp-baseline-") as temporary:
        baseline = Path(temporary)
        archive = subprocess.check_output(["git", "-C", str(repo), "archive", baseline_head])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # The archive comes only from the explicitly identified local commit.
            tar.extractall(baseline, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        paths = ["pdat_dsl/" + name for name in LEGACY_MODULES]
        paths.extend(str(path.relative_to(baseline)) for path in sorted((baseline / "configs").glob("*.yaml")))
        implementation = {path: {"baseline_sha256": sha((baseline / path).read_bytes()),
                                 "current_sha256": sha((repo / path).read_bytes())} for path in paths}
        for side, package in (("baseline", baseline), ("current", repo)):
            side_dir = output / side
            if side_dir.exists():
                raise ValueError("Use a fresh output directory: " + str(side_dir))
            env = dict(os.environ, PYTHONPATH=str(package), PYTHONHASHSEED=str(hash_seed))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(package), str(request_path), str(side_dir)],
                           cwd=temporary, env=env, check=True)
    before = json.loads((output / "baseline/results.json").read_text())
    after = json.loads((output / "current/results.json").read_text())
    mismatches = compare(before, after)
    source_changes = [name for name, hashes in implementation.items()
                      if hashes["baseline_sha256"] != hashes["current_sha256"]]
    input_changes = [] if corpus_snapshot else [name for name, item in inputs.items()
                     if not Path(item["path"]).exists() or sha(Path(item["path"]).read_bytes()) != item["sha256"]]
    errors = [dict(case=k, input=v["input"], mode=v["mode"], config=v["config"], error_type=v.get("error_type"), error=v.get("error"))
              for k, v in before.items() if v["status"] == "error"]
    generated = [v for v in before.values() if v["status"] == "generated"]
    timing_templates = []
    for case, result in before.items():
        for name, artifact in result["artifacts"].items():
            if name.endswith("_timing.sv"):
                unresolved = sorted(set(re.findall(r"\{(?:clk|rst_n|instr_req|instr_gnt|data_req|data_gnt|data_rvalid)\}",
                                                    Path(artifact["file"]).read_text())))
                if unresolved:
                    timing_templates.append({"case": case, "input": result["input"], "config": result["config"],
                                             "artifact": name, "unexpanded_names": unresolved})
    report = {
        "baseline_commit": baseline_head, "current_head": git(repo, "rev-parse", "HEAD"),
        "python": sys.version, "hash_seed": str(hash_seed), "inputs": inputs,
        "corpus_snapshot": str(corpus_snapshot) if corpus_snapshot else None,
        "live_input_mutation_check": not bool(corpus_snapshot),
        "dependencies": {name: importlib.metadata.version(name) for name in ("PyYAML", "jsonschema", "dd")},
        "legacy_implementation": implementation, "changed_legacy_implementation": source_changes,
        "inputs_changed_during_audit": input_changes,
        "input_count": len(inputs), "case_count": len(requests), "successful_cases": len(generated),
        "artifact_count": sum(len(v["artifacts"]) for v in generated),
        "nonempty_artifact_count": sum(a["bytes"] > 0 for v in generated for a in v["artifacts"].values()),
        "baseline_errors": errors, "mismatches": mismatches,
        "baseline_unexpanded_timing_templates": timing_templates,
        "passed": not mismatches and not source_changes and not input_changes,
        "limits": ["Compares generated SVA bytes; does not rerun synthesis, timing analysis, or processor verification.",
                   "Baseline generation failures are preserved and reported, not counted as successful generation.",
                   "Hierarchical operand checks exercise the actual emission API with a fixed explicit module binding at both 32 and 64 bits.",
                   "New examples/frontend golden pairs are checked separately by tests/test_frontend_compatibility.py.",
                   "Successful generation and byte parity do not establish valid HDL: inherited timing-template placeholders are reported separately.",
                   "Dynamic-scheduling experiments use independently written formal RTL; this audit does not claim to regenerate those artifacts."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("input_count", "case_count", "successful_cases", "artifact_count", "nonempty_artifact_count", "passed")}, indent=2))
    print("Baseline error cases:", len(errors), "Mismatches:", len(mismatches))
    print("Full report:", output / "report.json")
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=BASELINE)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for exact output files and JSON evidence")
    parser.add_argument("--corpus", type=Path, action="append", default=[], help="Additional Git repositories containing experiment DSLs")
    parser.add_argument("--corpus-snapshot", type=Path, help="Reproduce an exact input universe from a source/hash JSON snapshot")
    parser.add_argument("--hash-seed", default="0")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    extras = args.corpus or [repo.parent / "ScorrPdat", repo.parent / "BoundedDynamicPscorr"]
    if args.corpus and args.corpus_snapshot:
        parser.error("--corpus and --corpus-snapshot are mutually exclusive")
    return 0 if audit(repo, args.baseline_ref, args.output, extras, args.hash_seed, args.corpus_snapshot)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
