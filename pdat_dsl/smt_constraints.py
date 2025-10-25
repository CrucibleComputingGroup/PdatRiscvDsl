#!/usr/bin/env python3
"""
Generate SMT2 assumption constraints directly from DSL.

This bypasses SystemVerilog entirely and generates minimal SMT2
constraints on input signals.
"""

import sys
from pathlib import Path

from .parser import parse_dsl, InstructionRule, PatternRule, RequireRule, PcConstraintRule
from .codegen import instruction_rule_to_pattern


def generate_smt2_constraints(dsl_file: Path, module_name: str = "ibex_core") -> str:
    """
    Parse DSL and generate SMT2 constraints.

    Returns: SMT2 code that declares inputs and asserts constraints
    """

    # Parse DSL
    with open(dsl_file) as f:
        dsl_text = f.read()
    program = parse_dsl(dsl_text)

    smt2_lines = []
    smt2_lines.append("; Constraints generated from DSL")
    smt2_lines.append(f"; Source: {dsl_file.name}")
    smt2_lines.append("")

    # Check if any rule requires PC constraint
    has_pc_constraint = any(isinstance(rule, PcConstraintRule) for rule in program.rules)

    # Declare constrained inputs as free variables
    smt2_lines.append("; Cut-point inputs (free variables constrained by assumptions)")
    smt2_lines.append("(declare-const instr_rdata_i (_ BitVec 32))")
    smt2_lines.append("(declare-const rst_ni Bool)")
    smt2_lines.append("(declare-const instr_is_compressed Bool)")
    if has_pc_constraint:
        smt2_lines.append("(declare-const pc_if (_ BitVec 32))")
    smt2_lines.append("")

    # Generate constraints from DSL rules
    smt2_lines.append("; Instruction constraints from DSL")

    # Process each rule
    for rule in program.rules:
        if isinstance(rule, InstructionRule):
            # Outlawed instruction - convert to patterns
            patterns = instruction_rule_to_pattern(rule)

            for pattern, mask, desc in patterns:
                smt2_lines.append(f"; Outlawed: {desc}")
                smt2_lines.append(
                    f"(assert (or (not rst_ni) instr_is_compressed "
                    f"(not (= (bvand instr_rdata_i #x{mask:08x}) #x{pattern:08x}))))"
                )

        elif isinstance(rule, PatternRule):
            # Direct pattern constraint
            pattern = rule.pattern
            mask = rule.mask

            smt2_lines.append(f"; Pattern: {pattern:#010x} mask {mask:#010x}")
            smt2_lines.append(
                f"(assert (or (not rst_ni) instr_is_compressed "
                f"(not (= (bvand instr_rdata_i #x{mask:08x}) #x{pattern:08x}))))"
            )

        elif isinstance(rule, RequireRule):
            smt2_lines.append(f"; Required extension: {rule.extension}")
            # Positive constraints - would need to enumerate all valid instructions
            # Skip for now as negative constraints are sufficient

        elif isinstance(rule, PcConstraintRule):
            # PC address space constraint
            pc_bits = rule.pc_bits
            addr_space_kb = (2 ** pc_bits) // 1024
            smt2_lines.append(f"; PC constraint: {pc_bits}-bit address space ({addr_space_kb}KB)")
            smt2_lines.append(f"; Upper {32 - pc_bits} bits of PC must be 0")
            # Extract upper bits and assert they are zero
            # pc_if[31:pc_bits] == 0  ->  (bvand pc_if upper_mask) == 0
            # Upper mask: bits [31:pc_bits] set to 1, rest 0
            if pc_bits < 32:
                upper_mask = ((1 << (32 - pc_bits)) - 1) << pc_bits
                smt2_lines.append(
                    f"(assert (or (not rst_ni) "
                    f"(= (bvand pc_if #x{upper_mask:08x}) #x00000000)))"
                )

    smt2_lines.append("")
    return '\n'.join(smt2_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: dsl_to_smt_constraints.py <rules.dsl> [output.smt2]")
        print("")
        print("Generates SMT2 constraint assertions from DSL")
        sys.exit(1)

    dsl_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not dsl_file.exists():
        print(f"ERROR: DSL file not found: {dsl_file}")
        sys.exit(1)

    print("=" * 80)
    print("DSL → SMT2 CONSTRAINTS")
    print("=" * 80)
    print(f"Input: {dsl_file}")
    print()

    smt2_constraints = generate_smt2_constraints(dsl_file)

    if output_file:
        output_file.write_text(smt2_constraints)
        print(f"✓ Saved to: {output_file}")
    else:
        print(smt2_constraints)


if __name__ == '__main__':
    main()
