#!/usr/bin/env python3
"""
Convert VCD file to JSON state snapshot.

Extracts the final value of each signal from a VCD file and outputs as JSON.
Useful for converting the initial state VCD to a JSON format.

Usage: vcd_to_state_json.py <input.vcd> <output.json> [--strip-prefix PREFIX]
"""

import sys
import argparse
import json
from typing import Dict, List, Tuple
from collections import defaultdict

class VCDSignal:
    """Represents a signal from VCD file."""
    def __init__(self, identifier: str, name: str, width: int):
        self.identifier = identifier
        self.name = name
        self.width = width
        self.current_value = None

    def set_value(self, value: str):
        """Update current value."""
        self.current_value = value

def parse_vcd_header(lines: List[str]) -> Tuple[Dict[str, VCDSignal], int]:
    """Parse VCD header to extract signal declarations with full hierarchical paths."""
    signals = {}
    i = 0
    scope_stack = []

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('$enddefinitions'):
            return signals, i + 1

        # Parse scope entry
        if line.startswith('$scope'):
            parts = line.split()
            if len(parts) >= 3:
                scope_name = parts[2]
                scope_stack.append(scope_name)

        # Parse scope exit
        if line.startswith('$upscope'):
            if scope_stack:
                scope_stack.pop()

        # Parse variable declarations
        if line.startswith('$var'):
            parts = line.split()
            if len(parts) >= 5:
                var_type = parts[1]
                width = int(parts[2])
                identifier = parts[3]
                signal_name = parts[4]

                # Build full hierarchical name
                if scope_stack:
                    full_name = ".".join(scope_stack) + "." + signal_name
                else:
                    full_name = signal_name

                # Handle array indices like [31:0]
                if len(parts) > 5 and parts[5].startswith('['):
                    full_name += " " + parts[5]

                signals[identifier] = VCDSignal(identifier, full_name, width)

        i += 1

    return signals, len(lines)

def parse_vcd_data(lines: List[str], signals: Dict[str, VCDSignal], start_line: int):
    """Parse VCD data and update signal values to their final state."""
    current_time = 0

    for i in range(start_line, len(lines)):
        line = lines[i].strip()

        if not line or line.startswith('$'):
            continue

        # Timestamp
        if line.startswith('#'):
            try:
                current_time = int(line[1:])
            except ValueError:
                pass
            continue

        # Value change
        if line.startswith('b'):
            # Multi-bit value: b<value> <identifier>
            parts = line.split()
            if len(parts) >= 2:
                value = parts[0][1:]  # Remove 'b' prefix
                identifier = parts[1]
                if identifier in signals:
                    signals[identifier].set_value(value)
        else:
            # Single-bit value: <value><identifier>
            if len(line) >= 2:
                value = line[0]
                identifier = line[1:]
                if identifier in signals:
                    signals[identifier].set_value(value)

def strip_prefix_from_signals(signals: Dict[str, VCDSignal], prefix: str):
    """Strip a common prefix from all signal names."""
    if not prefix:
        return

    prefix_with_dot = prefix if prefix.endswith('.') else prefix + '.'

    for sig in signals.values():
        if sig.name.startswith(prefix_with_dot):
            sig.name = sig.name[len(prefix_with_dot):]

def export_state_json(signals: Dict[str, VCDSignal], output_file: str):
    """Export signal state to JSON."""

    # Group by bit-width for better organization
    state = {}

    for sig in signals.values():
        if sig.current_value is not None:
            state[sig.name] = {
                "width": sig.width,
                "value": sig.current_value
            }

    # Also create a flattened version for easy access
    output = {
        "signals": state,
        "summary": {
            "total_signals": len(state),
            "bit_width_distribution": {}
        }
    }

    # Count signals by width
    width_counts = defaultdict(int)
    for sig_data in state.values():
        width_counts[sig_data["width"]] += 1

    output["summary"]["bit_width_distribution"] = dict(sorted(width_counts.items()))

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description='Convert VCD file to JSON state snapshot',
        epilog='Example: vcd_to_state_json.py ibex_reset_state.vcd initial_state.json'
    )
    parser.add_argument('vcd_file', help='Input VCD file')
    parser.add_argument('output_json', help='Output JSON file')
    parser.add_argument('--strip-prefix', type=str, default='tb_ibex_random.dut',
                       help='Strip this prefix from signal paths (default: tb_ibex_random.dut)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print(f"Converting VCD to state JSON: {args.vcd_file}")
    print("")

    # Read VCD file
    try:
        with open(args.vcd_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: VCD file not found: {args.vcd_file}", file=sys.stderr)
        return 1

    # Parse header
    print("[1/4] Parsing VCD header...")
    signals, data_start = parse_vcd_header(lines)
    print(f"  Found {len(signals)} signals")
    print("")

    # Strip prefix
    print("[2/4] Stripping testbench prefix...")
    if args.strip_prefix:
        strip_prefix_from_signals(signals, args.strip_prefix)
        print(f"  Stripped prefix: {args.strip_prefix}")
    print("")

    # Parse data to get final values
    print("[3/4] Parsing VCD data to extract final values...")
    parse_vcd_data(lines, signals, data_start)

    # Count how many signals have values
    signals_with_values = sum(1 for s in signals.values() if s.current_value is not None)
    print(f"  Extracted values for {signals_with_values} signals")
    print("")

    # Export JSON
    print("[4/4] Exporting state to JSON...")
    export_state_json(signals, args.output_json)
    print(f"  Written: {args.output_json}")

    if args.verbose:
        # Show some example values
        print("\n  Example signal values:")
        count = 0
        for sig in signals.values():
            if sig.current_value is not None and count < 5:
                print(f"    {sig.name} ({sig.width}-bit) = {sig.current_value}")
                count += 1

    print("")
    print("=" * 80)
    print("State JSON Export Complete")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    exit(main())
