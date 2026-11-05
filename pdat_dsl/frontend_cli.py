"""Checked CRISP source elaboration with the unchanged experiment generator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .config import load_config, ModuleConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--config", type=Path)
    target.add_argument("--target", choices=["ibex", "boom", "rocket"])
    parser.add_argument("--check", action="store_true", help="Check without generating SVA")
    parser.add_argument("--emit-legacy", type=Path, help="Save the checked expansion")
    parser.add_argument("--manifest", type=Path, help="Save checks, bindings, and output hashes")
    parser.add_argument("--module-dtype", metavar="LABEL", help="Emit operand-only SVA with the selected config's signals; LABEL names the output scope")
    parser.add_argument("--legacy-parity", action="store_true", help="Explicitly reproduce standalone legacy output, including reported defects")
    args = parser.parse_args(argv)
    if not args.check and args.output is None:
        parser.error("output is required unless --check is used")
    paths = [p.resolve() for p in (args.input, args.output, args.emit_legacy, args.manifest, args.config) if p]
    if len(paths) != len(set(paths)):
        parser.error("input, output, expansion, and manifest must be different files")
    try:
        from .frontend import elaborate_frontend
        config = load_config(args.config, args.target)
        result = elaborate_frontend(args.input.read_text(), config, base_dir=args.input.parent)
        manifest = dict(result.manifest)
        imported_paths = {Path(entry["path"]).resolve() for entry in manifest.get("imports", [])}
        if any(p.resolve() in imported_paths for p in (args.output, args.emit_legacy, args.manifest) if p):
            raise ValueError("output or metadata would overwrite an imported definition library")
        if args.module_dtype and args.legacy_parity:
            raise ValueError("--module-dtype and --legacy-parity select different output routes")
        if args.module_dtype:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", args.module_dtype):
                raise ValueError("--module-dtype requires an identifier")
            if not result.manifest["rules"] or any(rule["role"] != "operand_assumption" for rule in result.manifest["rules"]):
                raise ValueError("--module-dtype requires only operand rules; it cannot emit ISA, register, PC, or timing assumptions")
            manifest["output_route"] = "existing hierarchical operand generator"
            manifest["legacy_effects_apply_to_output"] = False
            module = ModuleConfig(name=args.module_dtype, module_path=config.injection.module_path, signals=config.signals)
            manifest["module"] = {"name": module.name, "path": module.module_path,
                                  "mapping_source": "selected config.signals; label does not select config.modules"}
        else:
            for warning in manifest.get("warnings", []):
                print(f"CRISP frontend warning: {warning}", file=sys.stderr)
            if manifest.get("warnings") and not args.check and not args.legacy_parity:
                raise ValueError("standalone output has inherited defects; use an operand-only specification with --module-dtype NAME, or explicitly select --legacy-parity to reproduce the old artifacts")
            manifest["output_route"] = "standalone legacy codegen"
            manifest["legacy_effects_apply_to_output"] = True
        outputs = {}
        if not args.check and args.module_dtype:
            from .codegen import generate_module_dtype_assertions
            outputs[args.output] = generate_module_dtype_assertions(result.program.rules, module, config.data_width).encode()
        elif not args.check:
            with tempfile.TemporaryDirectory(prefix="crisp-frontend-") as directory:
                temporary = Path(directory)
                lowered = temporary / "expanded.dsl"
                lowered.write_text(result.legacy_text)
                output = temporary / "constraints.sv"
                command = [sys.executable, "-m", "pdat_dsl", "codegen", str(lowered), str(output)]
                if args.config:
                    command.extend(["--config", str(args.config.resolve())])
                elif args.target:
                    command.extend(["--target", args.target])
                run = subprocess.run(command, text=True, capture_output=True)
                if run.returncode:
                    raise ValueError("existing SVA generator failed:\n" + run.stdout + run.stderr)
                outputs[args.output] = output.read_bytes()
                timing = temporary / "constraints_timing.sv"
                if timing.exists():
                    name = str(args.output)
                    name = name[:-3] + "_timing.sv" if name.endswith(".sv") else name + "_timing.sv"
                    outputs[Path(name)] = timing.read_bytes()
            output_paths = [p.resolve() for p in outputs]
            protected = [p.resolve() for p in (args.input, args.emit_legacy, args.manifest, args.config) if p]
            protected.extend(imported_paths)
            if any(p in protected for p in output_paths):
                raise ValueError("a generated sidecar would overwrite an input or metadata file")
        for path, data in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest["sva_sha256"] = {str(p): hashlib.sha256(b).hexdigest() for p, b in outputs.items()}
        manifest["sva_backend"] = "unchanged generate_module_dtype_assertions" if args.module_dtype else "unchanged pdat_dsl codegen command"
        if args.emit_legacy:
            args.emit_legacy.parent.mkdir(parents=True, exist_ok=True)
            args.emit_legacy.write_text(result.legacy_text)
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print("CRISP frontend checks passed" + ("; generated SVA through the existing backend" if outputs else ""))
        return 0
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"CRISP frontend: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
