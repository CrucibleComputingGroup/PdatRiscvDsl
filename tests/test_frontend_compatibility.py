"""Exact-output checks against the pre-frontend generator, not a second emitter.

Golden sources are independent, explicit legacy counterparts of the reusable
frontend examples. Both interpreter seeds are matched between the frozen old
generator and the new entry point: the old generator itself has seed-dependent
instruction ordering. These checks make no processor-equivalence claim.
Timing sidecars intentionally retain inherited unexpanded template names;
matching bytes does not establish that those sidecars compile or bind to RTL.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "3fe0d81f0d8a2fed326fbd1d4095b1bd9f3b594a"
PAIRS = ("narrow", "timing", "ordered_isa", "fields")
CONFIGS = ("default", "ibex", "boom", "rocket", "riscvsinglecycle")


@pytest.fixture(scope="session")
def old_package(tmp_path_factory):
    destination = tmp_path_factory.mktemp("legacy-generator")
    archive = subprocess.check_output(["git", "archive", BASELINE], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    return destination


def run(package, args, seed, cwd):
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, text=True, capture_output=True,
        env=dict(os.environ, PYTHONPATH=str(package), PYTHONHASHSEED=str(seed)), timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def config_args(package, target):
    return [] if target == "default" else ["--config", str(package / "configs" / (target + ".yaml"))]


def sva_files(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.sv"))}


@pytest.mark.parametrize("seed", [0, 29])
@pytest.mark.parametrize("target", CONFIGS)
@pytest.mark.parametrize("name", PAIRS)
def test_frontend_cli_matches_frozen_generator(old_package, tmp_path, name, target, seed):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    expanded, manifest_file = new / "expanded.dsl", new / "manifest.json"
    run(old_package, ["-m", "pdat_dsl", "codegen", str(ROOT / "examples/frontend" / (name + ".dsl")),
                      str(old / "assumptions.sv"), *config_args(old_package, target)], seed, tmp_path)
    run(ROOT, ["-m", "pdat_dsl", "frontend", str(ROOT / "examples/frontend" / (name + ".crisp")),
               str(new / "assumptions.sv"), "--emit-legacy", str(expanded), "--manifest", str(manifest_file),
               *(["--legacy-parity"] if name in ("narrow", "timing") else []),
               *config_args(ROOT, target)], seed, tmp_path)
    assert sva_files(new) == sva_files(old)
    assert "assumptions.sv" in sva_files(new)
    assert ("assumptions_timing.sv" in sva_files(new)) == (name == "timing")
    manifest = json.loads(manifest_file.read_text())
    assert manifest["legacy_sha256"] == hashlib.sha256(expanded.read_bytes()).hexdigest()
    assert manifest["sva_sha256"] == {
        str(new / file): hashlib.sha256(data).hexdigest() for file, data in sva_files(new).items()
    }


@pytest.mark.parametrize("seed", [0, 29])
@pytest.mark.parametrize("width", [32, 64])
def test_operand_lowering_preserves_hierarchical_emission(old_package, tmp_path, width, seed):
    """Experiments also emit operand constraints through this separate API."""
    from pdat_dsl.config import load_config
    from pdat_dsl.frontend import elaborate_frontend

    source = ROOT / "examples/frontend/narrow.crisp"
    result = elaborate_frontend(source.read_text(), load_config(target="boom" if width == 64 else "ibex"), source.parent)
    expanded = tmp_path / "expanded.dsl"
    expanded.write_text(result.legacy_text)
    for side, package, input_file in (
        ("old", old_package, ROOT / "examples/frontend/narrow.dsl"),
        ("new", ROOT, expanded),
    ):
        requests = tmp_path / (side + "-requests.json")
        requests.write_text(json.dumps([{"input": "narrow", "mode": "hierarchical-dtype-api",
                                        "source": str(input_file), "config": str(width)}]))
        run(package, [str(ROOT / "scripts/check_sva_compatibility.py"), "--worker", str(package),
                      str(requests), str(tmp_path / side)], seed, tmp_path)
    before = (tmp_path / "old/0/module_dtype.sv").read_bytes()
    after = (tmp_path / "new/0/module_dtype.sv").read_bytes()
    assert before and before == after


@pytest.mark.parametrize("seed", [0, 29])
@pytest.mark.parametrize("target", CONFIGS)
def test_operand_only_cli_matches_frozen_module_api(old_package, tmp_path, target, seed):
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"input": "operands", "mode": "hierarchical-config-dtype-api",
                                    "source": str(ROOT / "examples/frontend/operands.dsl"), "config": target}]))
    run(old_package, [str(ROOT / "scripts/check_sva_compatibility.py"), "--worker", str(old_package),
                      str(requests), str(tmp_path / "old")], seed, tmp_path)
    run(ROOT, ["-m", "pdat_dsl", "frontend", str(ROOT / "examples/frontend/operands.crisp"),
               str(tmp_path / "module_dtype.sv"), "--module-dtype", "compat_ex",
               *config_args(ROOT, target)], seed, tmp_path)
    expected = (tmp_path / "old/0/module_dtype.sv").read_bytes()
    assert expected and (tmp_path / "module_dtype.sv").read_bytes() == expected


@pytest.mark.parametrize("name,extra", [
    ("narrow", []), ("timing", []), ("narrow", ["--module-dtype", "compat_ex"]),
])
def test_cli_cannot_silently_ban_operand_instruction_or_drop_rules(tmp_path, name, extra):
    output = tmp_path / "out.sv"
    result = subprocess.run(
        [sys.executable, "-m", "pdat_dsl", "frontend", str(ROOT / "examples/frontend" / (name + ".crisp")),
         str(output), *extra], cwd=tmp_path, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED="0"), timeout=60,
    )
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("collision", ["output", "expanded", "manifest", "sidecar"])
def test_cli_preserves_imported_library_at_every_output_location(tmp_path, collision):
    library = tmp_path / "out_timing.sv"
    library_bytes = b"crisp library 1\npredicate small(x: bits[W]) = fits_u(x, 8)\n"
    library.write_bytes(library_bytes)
    source = tmp_path / "source.crisp"
    source.write_text('crisp frontend 1\nimport "out_timing.sv"\nbackend 1\ncontext bus = memory\nat bus: data_hit_latency 1\n')
    output = library if collision == "output" else tmp_path / "out.sv"
    expanded = library if collision == "expanded" else tmp_path / "expanded.dsl"
    manifest = library if collision == "manifest" else tmp_path / "manifest.json"
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run(
        [sys.executable, "-m", "pdat_dsl", "frontend", str(source), str(output),
         "--legacy-parity", "--emit-legacy", str(expanded), "--manifest", str(manifest)],
        cwd=tmp_path, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED="0"), timeout=60,
    )
    assert result.returncode != 0 and "overwrite" in result.stderr
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


def test_module_check_reports_selected_mapping_without_emitting_sva(tmp_path):
    from pdat_dsl.config import load_config

    manifest = tmp_path / "manifest.json"
    run(ROOT, ["-m", "pdat_dsl", "frontend", str(ROOT / "examples/frontend/operands.crisp"),
               "--check", "--module-dtype", "review_scope", "--target", "boom", "--manifest", str(manifest)],
        0, tmp_path)
    info = json.loads(manifest.read_text())
    assert info["module"] == {
        "name": "review_scope", "path": load_config(target="boom").injection.module_path,
        "mapping_source": "selected config.signals; label does not select config.modules",
    }
    assert info["output_route"] == "existing hierarchical operand generator"
    assert info["legacy_effects_apply_to_output"] is False
    assert info["sva_sha256"] == {}
    assert not list(tmp_path.glob("*.sv"))


def test_audit_detects_changed_bytes_even_with_equal_recorded_hashes(tmp_path):
    """The audit must compare artifacts themselves, not trust JSON summaries."""
    spec = importlib.util.spec_from_file_location("compatibility_audit", ROOT / "scripts/check_sva_compatibility.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    left, right = tmp_path / "left.sv", tmp_path / "right.sv"
    left.write_text("assume (a);\n")
    right.write_text("assume (b);\n")
    def record(path):
        return {"0": {"status": "generated", "artifacts": {"a.sv": {"file": str(path), "sha256": "same"}}}}
    assert audit.compare(record(left), record(right)) == [
        {"case": "0", "reason": "SVA bytes changed", "artifact": "a.sv"}
    ]
