#!/usr/bin/env python3
"""
CLI entry point for pdat-dsl package.

Allows running the DSL tools via: python -m pdat_dsl <command>
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pdat-dsl",
        description="PDAT DSL - RISC-V ISA subset specification language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse and validate a DSL file
  python -m pdat_dsl parse examples/example_16reg.dsl

  # Generate SystemVerilog checker module
  python -m pdat_dsl codegen examples/example_16reg.dsl output.sv

  # Generate inline SystemVerilog assumptions
  python -m pdat_dsl codegen --inline examples/example_16reg.dsl output.sv

For more information, visit: https://github.com/yourusername/PdatDsl
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Parse command
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse and validate a DSL file"
    )
    parse_parser.add_argument("input_file", help="DSL file to parse")
    parse_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed parse output"
    )

    # Codegen command
    codegen_parser = subparsers.add_parser(
        "codegen",
        help="Generate SystemVerilog from DSL"
    )
    codegen_parser.add_argument("input_file", help="DSL file to process")
    codegen_parser.add_argument("output_file", help="Output SystemVerilog file")
    codegen_parser.add_argument(
        "--inline",
        action="store_true",
        help="Generate inline assumptions (not a module)"
    )
    codegen_parser.add_argument(
        "-m", "--module-name",
        default="instr_outlawed_checker",
        help="Name of generated module (default: instr_outlawed_checker)"
    )
    codegen_parser.add_argument(
        "-b", "--bind-file",
        help="Output bind file (default: <output_file_base>_bind.sv)"
    )
    
    # Timing constraint arguments
    codegen_parser.add_argument(
        "--timing",
        action="store_true",
        help="Enable cache-aware timing constraints"
    )
    codegen_parser.add_argument(
        "--timing-output",
        help="Output file for timing constraints (default: <output_base>_timing.sv)"
    )
    codegen_parser.add_argument(
        "--instr-hit-latency",
        type=int,
        default=1,
        help="Max cycles for instruction cache hit (default: 1)"
    )
    codegen_parser.add_argument(
        "--instr-miss-latency",
        type=int,
        default=5,
        help="Max cycles for instruction cache miss (default: 5)"
    )
    codegen_parser.add_argument(
        "--data-hit-latency",
        type=int,
        default=1,
        help="Max cycles for data cache hit (default: 1)"
    )
    codegen_parser.add_argument(
        "--data-miss-latency",
        type=int,
        default=4,
        help="Max cycles for data cache miss (default: 4)"
    )
    codegen_parser.add_argument(
        "--locality-bits",
        type=int,
        default=7,
        help="Number of address high bits for locality detection (default: 7)"
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Run built-in tests"
    )

    # VCD analysis commands
    vcd_state_parser = subparsers.add_parser(
        "vcd-to-state",
        help="Convert VCD to initial state JSON"
    )
    vcd_state_parser.add_argument("vcd_file", help="Input VCD file")
    vcd_state_parser.add_argument("output_json", help="Output JSON file")
    vcd_state_parser.add_argument("--strip-prefix", default="tb_ibex_random.dut",
                                   help="Strip prefix from signal names")
    vcd_state_parser.add_argument("-v", "--verbose", action="store_true")

    find_corr_parser = subparsers.add_parser(
        "find-correspondences",
        help="Find signal correspondences from VCD"
    )
    find_corr_parser.add_argument("vcd_file", help="Input VCD file")
    find_corr_parser.add_argument("output_json", help="Output JSON file")
    find_corr_parser.add_argument("--constants-only", action="store_true",
                                   help="Only constant signals")
    find_corr_parser.add_argument("--strip-prefix", default="tb_ibex_random.dut")
    find_corr_parser.add_argument("--report", help="Optional text report file")
    find_corr_parser.add_argument("-v", "--verbose", action="store_true")

    # SMT constraints command
    smt_parser = subparsers.add_parser(
        "smt-constraints",
        help="Generate SMT2 constraints from DSL"
    )
    smt_parser.add_argument("input_file", help="DSL file")
    smt_parser.add_argument("output_file", nargs="?", help="Output SMT2 file (optional)")

    # Random constraints command (for VCS testbenches)
    random_parser = subparsers.add_parser(
        "random-constraints",
        help="Generate SystemVerilog randomization constraints from DSL"
    )
    random_parser.add_argument("input_file", help="DSL file")
    random_parser.add_argument("output_file", help="Output SystemVerilog constraint class")
    random_parser.add_argument("-v", "--verbose", action="store_true")

    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )

    args = parser.parse_args()

    # Handle no command
    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to appropriate handler
    if args.command == "parse":
        from .parser import main as parser_main
        sys.argv = ["pdat-dsl", args.input_file]
        if args.verbose:
            sys.argv.append("-v")
        return parser_main()

    elif args.command == "codegen":
        from .codegen import main as codegen_main
        sys.argv = ["pdat-dsl", args.input_file, args.output_file]
        if args.inline:
            sys.argv.append("--inline")
        if args.module_name != "instr_outlawed_checker":
            sys.argv.extend(["-m", args.module_name])
        if args.bind_file:
            sys.argv.extend(["-b", args.bind_file])
        if args.timing:
            sys.argv.append("--timing")
        if args.timing_output:
            sys.argv.extend(["--timing-output", args.timing_output])
        if args.instr_hit_latency != 1:
            sys.argv.extend(["--instr-hit-latency", str(args.instr_hit_latency)])
        if args.instr_miss_latency != 5:
            sys.argv.extend(["--instr-miss-latency", str(args.instr_miss_latency)])
        if args.data_hit_latency != 1:
            sys.argv.extend(["--data-hit-latency", str(args.data_hit_latency)])
        if args.data_miss_latency != 4:
            sys.argv.extend(["--data-miss-latency", str(args.data_miss_latency)])
        if args.locality_bits != 7:
            sys.argv.extend(["--locality-bits", str(args.locality_bits)])
        return codegen_main()

    elif args.command == "test":
        from .parser import parse_dsl
        test_input = """
        # Test DSL
        require RV32I
        require_registers x0-x15
        instruction MUL
        instruction DIV
        """
        try:
            ast = parse_dsl(test_input)
            print(f"✓ Built-in test passed! Found {len(ast.rules)} rules")
            return 0
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return 1

    elif args.command == "vcd-to-state":
        from . import vcd_to_state_json
        sys.argv = ["pdat-dsl", args.vcd_file, args.output_json]
        if args.strip_prefix:
            sys.argv.extend(["--strip-prefix", args.strip_prefix])
        if args.verbose:
            sys.argv.append("-v")
        return vcd_to_state_json.main()

    elif args.command == "find-correspondences":
        from . import find_signal_correspondences
        sys.argv = ["pdat-dsl", args.vcd_file, args.output_json]
        if args.constants_only:
            sys.argv.append("--constants-only")
        if args.strip_prefix:
            sys.argv.extend(["--strip-prefix", args.strip_prefix])
        if args.report:
            sys.argv.extend(["--report", args.report])
        if args.verbose:
            sys.argv.append("-v")
        return find_signal_correspondences.main()

    elif args.command == "smt-constraints":
        from . import smt_constraints
        if args.output_file:
            sys.argv = ["pdat-dsl", args.input_file, args.output_file]
        else:
            sys.argv = ["pdat-dsl", args.input_file]
        return smt_constraints.main()

    elif args.command == "random-constraints":
        from . import random_constraints
        sys.argv = ["pdat-dsl", args.input_file, args.output_file]
        if args.verbose:
            sys.argv.append("-v")
        return random_constraints.main()

    elif args.command == "version":
        from . import __version__
        print(f"pdat-dsl version {__version__}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
