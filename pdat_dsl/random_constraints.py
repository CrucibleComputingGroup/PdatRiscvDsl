#!/usr/bin/env python3
"""
Generate SystemVerilog randomization constraints from instruction DSL.

This script:
1. Parses the DSL file containing instruction rules
2. Converts high-level instruction rules to SystemVerilog constraints
3. Generates a constraint class suitable for constrained randomization in VCS
"""

import sys
import argparse
from typing import List, Tuple, Set, Optional
from .parser import parse_dsl, InstructionRule, PatternRule, FieldConstraint, RequireRule, RegisterConstraintRule
from .encodings import (
    get_instruction_encoding, parse_register, get_extension_instructions,
    R_TYPE_FIELDS, I_TYPE_FIELDS, S_TYPE_FIELDS, B_TYPE_FIELDS
)

def get_format_register_fields(format_type: str) -> List[str]:
    """Return register fields for a given instruction format."""
    if format_type == 'R':
        return ['rd', 'rs1', 'rs2']
    elif format_type == 'I':
        # For I-type, check if it's a load (has rd, rs1) or other (may vary)
        return ['rd', 'rs1']
    elif format_type == 'S':
        return ['rs1', 'rs2']  # S-type has no rd
    elif format_type == 'B':
        return ['rs1', 'rs2']  # B-type has no rd
    elif format_type == 'U':
        return ['rd']
    elif format_type == 'J':
        return ['rd']
    return []

def generate_valid_instruction_constraint(required_extensions: Set[str]) -> str:
    """Generate constraint for valid instruction encodings from required extensions."""
    if not required_extensions:
        return ""

    # Collect all valid instruction patterns from required extensions
    valid_instrs = []
    for ext in sorted(required_extensions):
        ext_instrs = get_extension_instructions(ext)
        if ext_instrs:
            valid_instrs.extend(ext_instrs.values())

    if not valid_instrs:
        return ""

    # Generate constraint: instr must match one of the valid patterns
    constraint = "    // Valid instruction encodings from required extensions\n"
    constraint += "    constraint valid_encoding {\n"

    # Create a big OR of all valid patterns
    for i, instr in enumerate(valid_instrs):
        if i == 0:
            constraint += f"      ((instr_word & 32'h{instr.base_mask:08X}) == 32'h{instr.base_pattern:08X})"
        else:
            constraint += f" ||\n      ((instr_word & 32'h{instr.base_mask:08X}) == 32'h{instr.base_pattern:08X})"

    constraint += ";\n    }\n\n"
    return constraint

def generate_outlawed_instruction_constraints(patterns: List[Tuple[int, int, str]]) -> str:
    """Generate constraints to exclude outlawed instructions."""
    if not patterns:
        return ""

    constraint = "    // Outlawed instruction patterns\n"
    constraint += "    constraint no_outlawed_instrs {\n"

    for i, (pattern, mask, desc) in enumerate(patterns):
        constraint += f"      ((instr_word & 32'h{mask:08X}) != 32'h{pattern:08X})"

        if i < len(patterns) - 1:
            # Not the last item, add && for next line
            constraint += f" &&  // {desc}\n"
        else:
            # Last item, add semicolon
            constraint += f";  // {desc}\n"

    constraint += "    }\n\n"
    return constraint

def generate_register_constraints(reg_rules: List[RegisterConstraintRule], required_extensions: Set[str]) -> str:
    """
    Generate constraints for register fields.
    Format-aware: only constrains register fields that actually exist in the instruction format.
    """
    if not reg_rules:
        return ""

    # Take the first (most restrictive if multiple)
    reg_rule = reg_rules[0]
    min_reg = reg_rule.min_reg
    max_reg = reg_rule.max_reg

    constraint = f"    // Register constraints: x{min_reg}-x{max_reg}\n"

    # We need to check instruction format and only constrain relevant register fields
    # For each possible format, generate appropriate constraints

    # Get all instruction formats from required extensions
    all_formats = set()
    for ext in required_extensions:
        ext_instrs = get_extension_instructions(ext)
        if ext_instrs:
            for instr in ext_instrs.values():
                all_formats.add(instr.format)

    # Generate format-specific register constraints
    constraint += "    constraint valid_registers {\n"

    # Helper to check if instruction matches a format
    format_checks = {
        'R': "(instr_word[6:0] == 7'b0110011) || (instr_word[6:0] == 7'b0111011)",  # OP, OP-32
        'I': "(instr_word[6:0] == 7'b0010011) || (instr_word[6:0] == 7'b0000011) || (instr_word[6:0] == 7'b1100111) || (instr_word[6:0] == 7'b0011011)",  # OP-IMM, LOAD, JALR, OP-IMM-32
        'S': "(instr_word[6:0] == 7'b0100011)",  # STORE
        'B': "(instr_word[6:0] == 7'b1100011)",  # BRANCH
        'U': "(instr_word[6:0] == 7'b0110111) || (instr_word[6:0] == 7'b0010111)",  # LUI, AUIPC
        'J': "(instr_word[6:0] == 7'b1101111)",  # JAL
    }

    constraints_list = []

    for fmt in sorted(all_formats):
        reg_fields = get_format_register_fields(fmt)
        if not reg_fields:
            continue

        field_constraints = []
        for field in reg_fields:
            if field == 'rd':
                field_constraints.append(f"(instr_word[11:7] <= {max_reg})")
            elif field == 'rs1':
                field_constraints.append(f"(instr_word[19:15] <= {max_reg})")
            elif field == 'rs2':
                field_constraints.append(f"(instr_word[24:20] <= {max_reg})")

        if field_constraints:
            check = format_checks.get(fmt, "1'b0")
            combined = " && ".join(field_constraints)
            constraints_list.append(f"      (!({check}) || ({combined}))")

    if constraints_list:
        constraint += " &&\n".join(constraints_list)
        constraint += ";\n"
    else:
        constraint += "      1'b1;  // No format-specific constraints\n"

    constraint += "    }\n\n"
    return constraint

def instruction_rule_to_pattern(rule: InstructionRule) -> List[Tuple[int, int, str]]:
    """Convert an InstructionRule to one or more (pattern, mask, description) tuples."""
    encoding = get_instruction_encoding(rule.name)
    if not encoding:
        print(f"Warning: Unknown instruction '{rule.name}' at line {rule.line}, skipping", file=sys.stderr)
        return []

    pattern = encoding.base_pattern
    mask = encoding.base_mask

    # Apply field constraints
    for constraint in rule.constraints:
        field_name = constraint.field_name
        field_value = constraint.field_value

        if field_name not in encoding.fields:
            print(f"Warning: Field '{field_name}' not valid for {rule.name} at line {rule.line}", file=sys.stderr)
            continue

        field_pos, field_width = encoding.fields[field_name]

        # Handle wildcards
        if field_value in ('*', 'x', '_'):
            field_mask = (1 << field_width) - 1
            mask = mask & ~(field_mask << field_pos)
            continue

        # Handle register names
        if field_name in ('rd', 'rs1', 'rs2'):
            reg_num = parse_register(field_value)
            if reg_num is None:
                if isinstance(field_value, int):
                    reg_num = field_value
                else:
                    print(f"Warning: Invalid register '{field_value}' at line {rule.line}", file=sys.stderr)
                    continue
            field_value = reg_num

        # Handle numeric values
        if isinstance(field_value, str):
            try:
                if field_value.startswith('0x'):
                    field_value = int(field_value, 16)
                elif field_value.startswith('0b'):
                    field_value = int(field_value, 2)
                else:
                    field_value = int(field_value)
            except ValueError:
                print(f"Warning: Cannot parse field value '{field_value}' at line {rule.line}", file=sys.stderr)
                continue

        # Set the field in pattern and mask
        field_mask = (1 << field_width) - 1
        pattern = (pattern & ~(field_mask << field_pos)) | ((field_value & field_mask) << field_pos)
        mask = mask | (field_mask << field_pos)

    # Create description
    desc = f"{rule.name}"
    if rule.constraints:
        constraint_strs = [f"{c.field_name}={c.field_value}" for c in rule.constraints]
        desc += " { " + ", ".join(constraint_strs) + " }"

    return [(pattern, mask, desc)]

def generate_sv_class(
    required_extensions: Set[str],
    outlawed_patterns: List[Tuple[int, int, str]],
    register_rules: List[RegisterConstraintRule],
    class_name: str = "instr_constraints"
) -> str:
    """Generate SystemVerilog constraint class."""

    sv_code = f"""// Auto-generated instruction randomization constraints
// This class defines constraints for randomized instruction generation

class {class_name};
    // The instruction word to be randomized
    rand logic [31:0] instr_word;

"""

    # Generate valid instruction constraint
    if required_extensions:
        sv_code += generate_valid_instruction_constraint(required_extensions)

    # Generate outlawed instruction constraints
    if outlawed_patterns:
        sv_code += generate_outlawed_instruction_constraints(outlawed_patterns)

    # Generate register constraints
    if register_rules and required_extensions:
        sv_code += generate_register_constraints(register_rules, required_extensions)

    sv_code += "endclass\n"
    return sv_code

def main():
    parser = argparse.ArgumentParser(
        description='Generate SystemVerilog randomization constraints from DSL'
    )
    parser.add_argument('dsl_file', help='Input DSL file')
    parser.add_argument('output_file', help='Output SystemVerilog file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Read DSL file
    try:
        with open(args.dsl_file, 'r') as f:
            dsl_text = f.read()
    except FileNotFoundError:
        print(f"Error: DSL file '{args.dsl_file}' not found", file=sys.stderr)
        return 1

    # Parse DSL
    try:
        program = parse_dsl(dsl_text)
    except SyntaxError as e:
        print(f"Error parsing DSL: {e}", file=sys.stderr)
        return 1

    # Extract rules
    required_extensions = set()
    register_rules = []
    instruction_rules = []
    pattern_rules = []

    for rule in program.rules:
        if isinstance(rule, RequireRule):
            required_extensions.add(rule.extension)
        elif isinstance(rule, RegisterConstraintRule):
            register_rules.append(rule)
        elif isinstance(rule, InstructionRule):
            instruction_rules.append(rule)
        elif isinstance(rule, PatternRule):
            pattern_rules.append(rule)

    if args.verbose:
        print(f"Found {len(required_extensions)} required extensions: {sorted(required_extensions)}")
        print(f"Found {len(register_rules)} register constraint rules")
        print(f"Found {len(instruction_rules)} instruction rules")
        print(f"Found {len(pattern_rules)} pattern rules")

    # Convert instruction rules to patterns
    outlawed_patterns = []
    for rule in instruction_rules:
        patterns = instruction_rule_to_pattern(rule)
        outlawed_patterns.extend(patterns)

    # Add explicit pattern rules
    for rule in pattern_rules:
        desc = rule.description if rule.description else f"Pattern 0x{rule.pattern:08x}"
        outlawed_patterns.append((rule.pattern, rule.mask, desc))

    # Generate SystemVerilog
    sv_code = generate_sv_class(required_extensions, outlawed_patterns, register_rules)

    # Write output
    try:
        with open(args.output_file, 'w') as f:
            f.write(sv_code)
        print(f"Generated constraints: {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
