#!/usr/bin/env python3
"""
Find signal correspondences from VCD file using hash-based equivalence checking.

This script:
1. Parses VCD file to extract all signal values over time
2. Groups signals by bit-width
3. Hashes value sequences (including constant-0 and constant-1 patterns)
4. Finds hash collisions indicating potentially equivalent signals
5. Exports results as JSON

Usage: find_signal_correspondences.py <input.vcd> <output.json>
"""

import sys
import argparse
import json
import hashlib
from typing import Dict, List, Tuple, Set
from collections import defaultdict

class VCDSignal:
    """Represents a signal from VCD file."""
    def __init__(self, identifier: str, name: str, width: int, low_bit: int = 0):
        self.identifier = identifier  # VCD short identifier (e.g., "!")
        self.name = name              # Full hierarchical name
        self.width = width            # Bit width
        self.low_bit = low_bit        # Lowest bit index (e.g., 8 for [31:8])
        self.values = []              # List of (time, value) tuples

    def add_value(self, time: int, value: str):
        """Add a value change."""
        self.values.append((time, value))

    def get_hash(self) -> str:
        """Compute hash of value sequence."""
        # Create a string representation of all values
        value_str = "|".join([f"{t}:{v}" for t, v in self.values])
        return hashlib.sha256(value_str.encode()).hexdigest()

    def is_constant(self) -> bool:
        """Check if signal has constant value (never changes)."""
        if len(self.values) == 0:
            return True
        if len(self.values) == 1:
            return True
        # Check if all values are the same
        first_val = self.values[0][1]
        return all(v[1] == first_val for v in self.values)

    def get_constant_value(self) -> str:
        """Get constant value if signal is constant, None otherwise."""
        if self.is_constant() and len(self.values) > 0:
            return self.values[0][1]
        return None

    def get_constant_bits(self, low_bit: int = 0) -> Dict[int, str]:
        """
        Find which individual bits are constant (for multi-bit signals).

        Args:
            low_bit: The lowest bit index of the signal (e.g., 8 for [31:8], 0 for [7:0])

        Returns: Dict[actual_bit_index, constant_value]
        Example: {4: '0', 7: '1'} means hardware bit 4 is always 0, bit 7 is always 1
        """
        if self.width == 1:
            # For 1-bit signals, use is_constant()
            return {}

        if len(self.values) == 0:
            return {}

        constant_bits = {}

        # Check each bit position in the value string
        for value_idx in range(self.width):
            bit_values = []

            for time, value in self.values:
                # Extract bit at position value_idx in the string
                # LSB is rightmost, so value_idx 0 is value[-1], value_idx 1 is value[-2], etc.
                if len(value) > value_idx:
                    bit_val = value[-(value_idx + 1)]
                    bit_values.append(bit_val)

            # Check if this bit is constant
            if bit_values and all(b == bit_values[0] for b in bit_values):
                # Map value_idx to actual hardware bit index
                actual_bit_idx = low_bit + value_idx
                constant_bits[actual_bit_idx] = bit_values[0]

        return constant_bits

def parse_vcd_header(lines: List[str]) -> Tuple[Dict[str, VCDSignal], int]:
    """
    Parse VCD header to extract signal declarations with full hierarchical paths.
    Returns: (signal_dict, header_end_line)
    """
    signals = {}
    i = 0
    timescale = "1ns"

    # Track hierarchy with scope stack
    scope_stack = []

    while i < len(lines):
        line = lines[i].strip()

        # Check for end of header
        if line.startswith('$enddefinitions'):
            return signals, i + 1

        # Parse timescale
        if line.startswith('$timescale'):
            timescale = lines[i+1].strip() if i+1 < len(lines) else "1ns"

        # Parse scope entry
        if line.startswith('$scope'):
            parts = line.split()
            if len(parts) >= 3:
                # Format: $scope <type> <name> $end
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
                # Format: $var <type> <width> <identifier> <name> [range] $end
                var_type = parts[1]
                width = int(parts[2])
                identifier = parts[3]
                signal_name = parts[4]

                # Build full hierarchical name
                if scope_stack:
                    full_name = ".".join(scope_stack) + "." + signal_name
                else:
                    full_name = signal_name

                # Parse bit range to get low_bit (e.g., [31:8] -> low_bit=8)
                low_bit = 0
                if len(parts) > 5 and parts[5].startswith('['):
                    full_name += " " + parts[5]
                    # Extract bit range
                    import re
                    bit_range_match = re.search(r'\[(\d+):(\d+)\]', parts[5])
                    if bit_range_match:
                        high = int(bit_range_match.group(1))
                        low = int(bit_range_match.group(2))
                        low_bit = low

                signals[identifier] = VCDSignal(identifier, full_name, width, low_bit)

        i += 1

    return signals, len(lines)

def strip_prefix_from_signals(signals: Dict[str, VCDSignal], prefix: str):
    """Strip a common prefix from all signal names."""
    if not prefix:
        return

    prefix_with_dot = prefix if prefix.endswith('.') else prefix + '.'

    for sig in signals.values():
        if sig.name.startswith(prefix_with_dot):
            sig.name = sig.name[len(prefix_with_dot):]

def parse_vcd_data(lines: List[str], signals: Dict[str, VCDSignal], start_line: int):
    """Parse VCD data section and populate signal values."""
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
        # Format: "bVALUE IDENTIFIER" or "VALUE+IDENTIFIER" (single bit)
        if line.startswith('b'):
            # Multi-bit value: b<value> <identifier>
            parts = line.split()
            if len(parts) >= 2:
                value = parts[0][1:]  # Remove 'b' prefix
                identifier = parts[1]
                if identifier in signals:
                    signals[identifier].add_value(current_time, value)
        else:
            # Single-bit value: <value><identifier>
            if len(line) >= 2:
                value = line[0]
                identifier = line[1:]
                if identifier in signals:
                    signals[identifier].add_value(current_time, value)

def create_constant_signal(width: int, value: int, name: str) -> VCDSignal:
    """Create a constant signal (all 0s or all 1s)."""
    sig = VCDSignal(f"_const_{name}_{width}", f"CONSTANT_{name}", width)

    # Create constant value string
    if value == 0:
        const_value = "0" * width if width > 1 else "0"
    else:
        const_value = "1" * width if width > 1 else "1"

    # Add a single value (constants don't change)
    sig.add_value(0, const_value)

    return sig

def group_signals_by_width(signals: Dict[str, VCDSignal]) -> Dict[int, List[VCDSignal]]:
    """Group signals by their bit width."""
    groups = defaultdict(list)

    for sig in signals.values():
        groups[sig.width].append(sig)

    return dict(groups)

def find_arbitrary_constants(signals: List[VCDSignal]) -> List[Tuple[str, str, int]]:
    """
    Find all constant signals (including all-0s and all-1s).

    Returns: List of (signal_name, constant_value, width)
    """
    arbitrary_constants = []

    for sig in signals:
        if sig.is_constant():
            const_val = sig.get_constant_value()
            if const_val is None:
                continue

            # Include ALL constant signals, not just arbitrary ones
            # This is important for 1-bit signals which can only be 0 or 1
            arbitrary_constants.append((sig.name, const_val, sig.width))

    return arbitrary_constants


def find_per_bit_constants(signals: List[VCDSignal]) -> List[Tuple[str, int, str]]:
    """
    Find individual constant bits within multi-bit signals.

    For signals where the full signal is NOT constant, but individual bits are.
    Example: rf_raddr[4:0] might vary, but rf_raddr[4] is always 0.

    Returns: List of (signal_name, bit_index, bit_value)
    """
    per_bit_constants = []

    for sig in signals:
        # Skip signals that are fully constant (already handled)
        if sig.is_constant():
            continue

        # Skip 1-bit signals (no individual bits to check)
        if sig.width <= 1:
            continue

        # Check each bit (pass low_bit to get actual hardware bit indices)
        constant_bits = sig.get_constant_bits(sig.low_bit)

        for bit_idx, bit_value in constant_bits.items():
            per_bit_constants.append((sig.name, bit_idx, bit_value))

    return per_bit_constants

def find_correspondences(signals_by_width: Dict[int, List[VCDSignal]]) -> Tuple[Dict[int, Dict[str, List[str]]], Dict[int, List[Tuple[str, str]]], List[Tuple[str, int, str]]]:
    """
    Find signal correspondences by hash collision.

    Returns: (correspondences, arbitrary_constants, per_bit_constants)
        correspondences: {width: {hash: [signal_names]}}
        arbitrary_constants: {width: [(signal_name, constant_value)]}
        per_bit_constants: [(signal_name, bit_index, bit_value)]
    """
    correspondences = {}
    all_arbitrary_constants = {}
    all_per_bit_constants = []

    for width, signals in signals_by_width.items():
        # Find arbitrary constants for this width
        arb_consts = find_arbitrary_constants(signals)
        if arb_consts:
            all_arbitrary_constants[width] = [(name, val) for name, val, w in arb_consts]

        # Find per-bit constants
        per_bit_consts = find_per_bit_constants(signals)
        all_per_bit_constants.extend(per_bit_consts)

        # Add constant signals
        const_zero = create_constant_signal(width, 0, "ZERO")
        const_ones = create_constant_signal(width, 1, "ONES")
        all_signals = signals + [const_zero, const_ones]

        # Hash all signals
        hash_to_signals = defaultdict(list)

        for sig in all_signals:
            sig_hash = sig.get_hash()
            hash_to_signals[sig_hash].append(sig.name)

        # Only keep hashes with multiple signals (collisions)
        collisions = {h: names for h, names in hash_to_signals.items() if len(names) > 1}

        if collisions:
            correspondences[width] = collisions

    return correspondences, all_arbitrary_constants, all_per_bit_constants

def format_correspondences_report(correspondences: Dict[int, Dict[str, List[str]]],
                                   arbitrary_constants: Dict[int, List[Tuple[str, str]]],
                                   constants_only: bool = False) -> str:
    """Format correspondences as human-readable report."""
    report = []
    report.append("=" * 80)
    report.append("Signal Correspondence Analysis Report")
    if constants_only:
        report.append("(Constants-Only Mode: CONSTANT_ZERO/ONES equivalences)")
    report.append("=" * 80)
    report.append("")

    total_groups = 0
    total_signals_in_groups = 0

    for width in sorted(correspondences.keys()):
        collisions = correspondences[width]
        report.append(f"Bit-width {width}:")
        report.append("-" * 40)

        for i, (hash_val, signal_names) in enumerate(sorted(collisions.items()), 1):
            # Identify if this is a constant class
            is_const_zero = "CONSTANT_ZERO" in signal_names
            is_const_ones = "CONSTANT_ONES" in signal_names

            class_label = f"  Group {i} ({len(signal_names)} signals)"
            if is_const_zero:
                class_label += " ≡ CONSTANT_ZERO"
            elif is_const_ones:
                class_label += " ≡ CONSTANT_ONES"

            report.append(class_label + ":")
            for name in sorted(signal_names):
                report.append(f"    - {name}")
            report.append("")

            total_groups += 1
            total_signals_in_groups += len(signal_names)

        report.append("")

    report.append("=" * 80)
    report.append(f"Summary:")
    report.append(f"  Total equivalence groups: {total_groups}")
    report.append(f"  Total signals in groups: {total_signals_in_groups}")

    if arbitrary_constants:
        total_arb = sum(len(consts) for consts in arbitrary_constants.values())
        report.append(f"  Total arbitrary constants: {total_arb}")

    report.append("=" * 80)

    return "\n".join(report)

def export_json(correspondences: Dict[int, Dict[str, List[str]]],
                arbitrary_constants: Dict[int, List[Tuple[str, str]]],
                per_bit_constants: List[Tuple[str, int, str]],
                output_file: str,
                constants_only: bool = False):
    """Export correspondences to JSON format."""

    # Convert to more readable JSON structure
    output = {
        "equivalence_classes": [],
        "arbitrary_constants": [],
        "per_bit_constants": []
    }

    class_id = 0
    for width in sorted(correspondences.keys()):
        collisions = correspondences[width]

        for hash_val, signal_names in collisions.items():
            # Check if this is a constant class
            is_const_zero = "CONSTANT_ZERO" in signal_names
            is_const_ones = "CONSTANT_ONES" in signal_names
            is_constant_class = is_const_zero or is_const_ones

            # Filter if constants_only mode
            if constants_only and not is_constant_class:
                continue

            output["equivalence_classes"].append({
                "class_id": class_id,
                "bit_width": width,
                "signal_count": len(signal_names),
                "signals": sorted(signal_names),
                "hash": hash_val[:16],  # Truncated hash for readability
                "is_constant_zero": is_const_zero,
                "is_constant_ones": is_const_ones
            })
            class_id += 1

    # Add arbitrary constants
    for width in sorted(arbitrary_constants.keys()):
        for signal_name, const_value in arbitrary_constants[width]:
            output["arbitrary_constants"].append({
                "signal": signal_name,
                "bit_width": width,
                "value": const_value
            })

    # Add per-bit constants
    for signal_name, bit_index, bit_value in per_bit_constants:
        output["per_bit_constants"].append({
            "signal": signal_name,
            "bit_index": bit_index,
            "bit_value": bit_value
        })

    output["summary"] = {
        "total_equivalence_classes": len(output["equivalence_classes"]),
        "total_signals_in_classes": sum(len(ec["signals"]) for ec in output["equivalence_classes"]),
        "total_arbitrary_constants": len(output["arbitrary_constants"]),
        "total_per_bit_constants": len(output["per_bit_constants"]),
        "constants_only_filter": constants_only
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description='Find signal correspondences in VCD file via hash-based equivalence',
        epilog='Example: find_signal_correspondences.py sim.vcd output.json --report report.txt'
    )
    parser.add_argument('vcd_file', help='Input VCD file')
    parser.add_argument('output_json', help='Output JSON file with correspondences')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--report', help='Optional: Write human-readable text report')
    parser.add_argument('--constants-only', action='store_true',
                       help='Only output signals equivalent to CONSTANT_ZERO or CONSTANT_ONES')
    parser.add_argument('--strip-prefix', type=str, default='tb_ibex_random.dut',
                       help='Strip this prefix from signal paths (default: tb_ibex_random.dut)')
    parser.add_argument('--min-group-size', type=int, default=2,
                       help='Minimum signals per group (default: 2)')

    args = parser.parse_args()

    print(f"Analyzing VCD file: {args.vcd_file}")
    print("")

    # Read VCD file
    try:
        with open(args.vcd_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: VCD file not found: {args.vcd_file}", file=sys.stderr)
        return 1

    # Parse header
    print("[1/5] Parsing VCD header...")
    signals, data_start = parse_vcd_header(lines)
    print(f"  Found {len(signals)} signals")

    if args.verbose:
        width_counts = defaultdict(int)
        for sig in signals.values():
            width_counts[sig.width] += 1
        print("  Signal width distribution:")
        for width in sorted(width_counts.keys()):
            print(f"    {width}-bit: {width_counts[width]} signals")
    print("")

    # Strip prefix
    print("[2/5] Stripping testbench prefix from signal names...")
    if args.strip_prefix:
        strip_prefix_from_signals(signals, args.strip_prefix)
        print(f"  Stripped prefix: {args.strip_prefix}")
    else:
        print(f"  No prefix stripping")
    print("")

    # Parse data
    print("[3/5] Parsing VCD data...")
    parse_vcd_data(lines, signals, data_start)
    print(f"  Parsed value changes for {len(signals)} signals")
    print("")

    # Group by width
    print("[4/5] Grouping signals by width and computing hashes...")
    signals_by_width = group_signals_by_width(signals)

    # Find correspondences, arbitrary constants, and per-bit constants
    correspondences, arbitrary_constants, per_bit_constants = find_correspondences(signals_by_width)

    num_groups = sum(len(colls) for colls in correspondences.values())
    num_arb_consts = sum(len(consts) for consts in arbitrary_constants.values())
    print(f"  Found {num_groups} equivalence groups")
    print(f"  Found {num_arb_consts} arbitrary constant signals")
    print(f"  Found {len(per_bit_constants)} per-bit constants")
    print("")

    # Export results
    print("[5/5] Exporting results...")
    export_json(correspondences, arbitrary_constants, per_bit_constants, args.output_json, args.constants_only)
    print(f"  JSON: {args.output_json}")
    if args.constants_only:
        print(f"  (constants-only mode: filtered for CONSTANT_ZERO/ONES equivalences)")

    # Optional text report
    if args.report:
        report_text = format_correspondences_report(correspondences, arbitrary_constants, args.constants_only)

        # Add detailed arbitrary constants section
        if arbitrary_constants:
            report_text += "\n\n" + "=" * 80 + "\n"
            report_text += "ARBITRARY CONSTANT SIGNALS\n"
            report_text += "(Constant values that are neither all-0s nor all-1s)\n"
            report_text += "=" * 80 + "\n\n"

            for width in sorted(arbitrary_constants.keys()):
                report_text += f"Bit-width {width}:\n"
                report_text += "-" * 40 + "\n"
                for signal_name, const_value in arbitrary_constants[width]:
                    report_text += f"  {signal_name}\n"
                    report_text += f"    Value: {const_value} ({width}'b{const_value})\n"
                report_text += "\n"

        with open(args.report, 'w') as f:
            f.write(report_text)
        print(f"  Report: {args.report}")

    print("")
    print("=" * 80)
    print("Signal Correspondence Analysis Complete")
    print("=" * 80)

    # Show quick summary
    for width in sorted(correspondences.keys())[:5]:  # Show first 5 widths
        num_classes = len(correspondences[width])
        print(f"  {width}-bit signals: {num_classes} equivalence classes")

    if len(correspondences) > 5:
        print(f"  ... and {len(correspondences) - 5} more bit-widths")

    if num_arb_consts > 0:
        print(f"\n  Arbitrary constants: {num_arb_consts} signals")

    return 0

if __name__ == '__main__':
    exit(main())
