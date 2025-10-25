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

  # Generate inline SystemVerilog assumptions (default: Ibex core)
  python -m pdat_dsl codegen examples/example_16reg.dsl output.sv

  # Generate for a specific core
  python -m pdat_dsl codegen --target boom examples/example.dsl output.sv

  # Use custom core configuration
  python -m pdat_dsl codegen --config mycore.yaml examples/example.dsl output.sv

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
        help="Generate inline SystemVerilog assumptions from DSL"
    )
    codegen_parser.add_argument("input_file", help="DSL file to process")
    codegen_parser.add_argument("output_file", help="Output SystemVerilog file (inline assumptions)")
    codegen_parser.add_argument(
        "--config",
        type=Path,
        help="Core configuration YAML file (default: builtin Ibex config)"
    )
    codegen_parser.add_argument(
        "--target",
        choices=["ibex", "boom", "rocket"],
        help="Target core shortcut (ibex, boom, rocket)"
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
        if hasattr(args, 'config') and args.config:
            sys.argv.extend(["--config", str(args.config)])
        if hasattr(args, 'target') and args.target:
            sys.argv.extend(["--target", args.target])
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

    elif args.command == "version":
        from . import __version__
        print(f"pdat-dsl version {__version__}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
