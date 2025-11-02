#!/usr/bin/env python3
"""
Generate inline SystemVerilog assumptions from instruction DSL.

This script:
1. Parses the DSL file containing instruction rules
2. Converts high-level instruction rules to pattern/mask pairs
3. Generates inline SystemVerilog assumptions (no separate module)
4. Generates data type constraint assertions for operand bit patterns

Output is designed to be injected directly into the target core's ID/decode stage.
"""

import sys
import argparse
from typing import List, Tuple, Set, Optional
from pathlib import Path
from .parser import parse_dsl, InstructionRule, PatternRule, FieldConstraint, RequireRule, RegisterConstraintRule, PcConstraintRule, DataTypeSet, BitPattern, TimingConstraintRule, IncludeRule, ForbidRule, InstructionPattern, RangeValue
from .encodings import (
    get_instruction_encoding, parse_register, set_field, create_field_mask,
    get_extension_instructions
)
from .dtype_codegen import generate_dtype_check_expr, generate_dtype_set_check_expr
from .config import CoreConfig, ModuleConfig, SignalConfig, load_config
from .bdd_ops import (
    get_bdd_manager, pattern_to_bdd, field_range_to_bdd, field_equals_to_bdd,
    bdd_to_patterns, bdd_size
)


def has_c_extension_required(rules: List) -> bool:
    """Check if RV32C or RV64C extension is required in the rules."""
    for rule in rules:
        if isinstance(rule, RequireRule):
            if rule.extension.upper() in ('RV32C', 'RV64C'):
                return True
    return False


def generate_complementary_patterns(allowed_pattern: int, allowed_mask: int, base_pattern: int, base_mask: int, field_pos: int, field_width: int) -> List[Tuple[int, int, str]]:
    """
    Generate patterns that forbid everything EXCEPT the allowed pattern.

    For an 'allow' rule, we need to generate patterns that outlaw all other values
    in the constrained field while keeping the rest of the instruction encoding the same.

    Args:
        allowed_pattern: The pattern value that IS allowed
        allowed_mask: The mask for the allowed pattern (which bits are constrained)
        base_pattern: The instruction's base pattern
        base_mask: The instruction's base mask
        field_pos: Position of the field in the instruction
        field_width: Width of the field in bits

    Returns:
        List of (pattern, mask, description) tuples that forbid all values except allowed
    """
    complementary = []

    # For each possible value in the field
    max_value = (1 << field_width) - 1
    for value in range(max_value + 1):
        # Check if this value matches the allowed pattern
        if (value & (allowed_mask >> field_pos)) == ((allowed_pattern >> field_pos) & (allowed_mask >> field_pos)):
            # This value is allowed, skip it
            continue

        # This value should be forbidden - create a pattern for it
        forbidden_pattern = base_pattern | (value << field_pos)
        forbidden_mask = base_mask | ((1 << field_width) - 1) << field_pos  # Check all bits of the field

        complementary.append((forbidden_pattern, forbidden_mask, f"complement value {value}"))

    return complementary


def expand_instruction_pattern_v2_bdd(pattern: InstructionPattern, use_bdd: bool = False):
    """
    Expand an InstructionPattern to a BDD or set of tuples.

    Args:
        pattern: InstructionPattern with name and constraints
        use_bdd: If True, return BDD; if False, return Set[Tuple] (legacy)

    Returns:
        BDD if use_bdd=True, otherwise Set[Tuple[int, int, str, bool]]
    """
    # Handle wildcard - matches all instructions (or all with constraints)
    if pattern.name == "*":
        width = 32  # Default to 32-bit for wildcard
        bdd_mgr = get_bdd_manager(width)

        # Start with TRUE (all instructions)
        base_bdd = bdd_mgr.true

        # Apply field constraints to narrow down
        # For wildcard, we just apply the constraints without a base pattern
        for constraint in pattern.constraints:
            field_name = constraint.field_name
            field_value = constraint.field_value

            # For wildcard, assume standard RISC-V register field positions
            # rd: [11:7], rs1: [19:15], rs2: [24:20]
            field_map = {
                "rd": (7, 5),
                "rs1": (15, 5),
                "rs2": (20, 5),
            }

            if field_name not in field_map:
                print(f"Warning: Field '{field_name}' not supported for wildcard patterns (use rd, rs1, rs2)")
                continue

            field_pos, field_width = field_map[field_name]

            # Handle range constraints
            if isinstance(field_value, RangeValue):
                range_bdd = field_range_to_bdd(field_pos, field_width,
                                               field_value.min_val, field_value.max_val, width)
                base_bdd &= range_bdd
                continue

        if use_bdd:
            return base_bdd

        # Convert to patterns (for wildcard, this might be many patterns)
        patterns = bdd_to_patterns(base_bdd, width, max_patterns=10000)
        result = set()
        for p, m, _ in patterns:
            result.add((p, m, f"* {pattern.constraints}", False))
        return result

    # Regular instruction (non-wildcard)
    encoding = get_instruction_encoding(pattern.name)
    if not encoding:
        print(f"Warning: Unknown instruction '{pattern.name}', skipping")
        if use_bdd:
            bdd = get_bdd_manager(32)
            return bdd.false
        return set()

    width = 16 if encoding.is_compressed else 32
    bdd_mgr = get_bdd_manager(width)

    # Start with base instruction pattern as BDD
    base_bdd = pattern_to_bdd(encoding.base_pattern, encoding.base_mask, width)

    # Apply each constraint
    for constraint in pattern.constraints:
        field_name = constraint.field_name
        field_value = constraint.field_value

        # Skip dtype constraints
        if field_name == 'dtype' or field_name.endswith('_dtype'):
            continue

        if field_name not in encoding.fields:
            print(f"Warning: Field '{field_name}' not valid for {pattern.name}")
            continue

        field_pos, field_width = encoding.fields[field_name]

        # Handle range constraints (NEW with BDD!)
        if isinstance(field_value, RangeValue):
            range_bdd = field_range_to_bdd(field_pos, field_width,
                                           field_value.min_val, field_value.max_val, width)
            base_bdd &= range_bdd
            continue

        # Handle bit patterns
        if isinstance(field_value, BitPattern):
            if field_value.width != field_width:
                print(f"Error: Bit pattern width {field_value.width} != field width {field_width}")
                continue

            pattern_val, pattern_mask = field_value.to_pattern_mask()
            # Create BDD for this bit pattern
            pattern_bdd = pattern_to_bdd(pattern_val << field_pos, pattern_mask << field_pos, width)
            base_bdd &= pattern_bdd
            continue

        # Handle exact values (wildcards, registers, numbers)
        if field_value in ('*', 'x', '_'):
            # Wildcard - no constraint, BDD unchanged
            continue

        # Handle registers
        if field_name in ('rd', 'rs1', 'rs2'):
            reg_num = parse_register(field_value)
            if reg_num is None:
                if isinstance(field_value, int):
                    reg_num = field_value
                else:
                    print(f"Warning: Invalid register '{field_value}'")
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
                print(f"Warning: Cannot parse field value '{field_value}'")
                continue

        # Create BDD for field == value
        eq_bdd = field_equals_to_bdd(field_pos, field_width, field_value, width)
        base_bdd &= eq_bdd

    if use_bdd:
        return base_bdd

    # Legacy: Convert BDD back to pattern set
    patterns = bdd_to_patterns(base_bdd, width)
    desc = f"{pattern.name}"
    if pattern.constraints:
        constraint_strs = [f"{c.field_name}={c.field_value}" for c in pattern.constraints]
        desc += " { " + ", ".join(constraint_strs) + " }"

    result = set()
    for p, m, _ in patterns:
        result.add((p, m, desc, encoding.is_compressed))
    return result


def expand_instruction_pattern_v2(pattern: InstructionPattern) -> Set[Tuple[int, int, str, bool]]:
    """
    Expand an InstructionPattern to a set of (pattern, mask, description, is_compressed) tuples.

    For v2, this generates all instruction instances matching the pattern constraints.
    Similar to instruction_rule_to_pattern but returns a set and handles v2 semantics.
    """
    encoding = get_instruction_encoding(pattern.name)
    if not encoding:
        print(f"Warning: Unknown instruction '{pattern.name}', skipping")
        return set()

    # Start with base pattern/mask
    base_pattern = encoding.base_pattern
    base_mask = encoding.base_mask

    # Apply constraints
    for constraint in pattern.constraints:
        field_name = constraint.field_name
        field_value = constraint.field_value

        # Skip dtype constraints
        if field_name == 'dtype' or field_name.endswith('_dtype'):
            continue

        if field_name not in encoding.fields:
            print(f"Warning: Field '{field_name}' not valid for {pattern.name}")
            continue

        field_pos, field_width = encoding.fields[field_name]

        # Handle bit patterns
        if isinstance(field_value, BitPattern):
            if field_value.width != field_width:
                print(f"Error: Bit pattern width {field_value.width} != field width {field_width}")
                continue

            pattern_val, pattern_mask = field_value.to_pattern_mask()
            base_pattern = set_field(base_pattern, field_pos, field_width, pattern_val)
            base_mask = base_mask | (pattern_mask << field_pos)
            continue

        # Handle wildcards
        if field_value in ('*', 'x', '_'):
            field_mask = create_field_mask(field_pos, field_width)
            base_mask = base_mask & ~field_mask
            continue

        # Handle registers
        if field_name in ('rd', 'rs1', 'rs2'):
            reg_num = parse_register(field_value)
            if reg_num is None:
                if isinstance(field_value, int):
                    reg_num = field_value
                else:
                    print(f"Warning: Invalid register '{field_value}'")
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
                print(f"Warning: Cannot parse field value '{field_value}'")
                continue

        base_pattern = set_field(base_pattern, field_pos, field_width, field_value)
        base_mask = base_mask | create_field_mask(field_pos, field_width)

    desc = f"{pattern.name}"
    if pattern.constraints:
        constraint_strs = [f"{c.field_name}={c.field_value}" for c in pattern.constraints]
        desc += " { " + ", ".join(constraint_strs) + " }"

    return {(base_pattern, base_mask, desc, encoding.is_compressed)}


def expand_expr_v2_bdd(expr, has_c_ext: bool = False, width: int = 32):
    """
    Expand a v2 expression to a BDD representing the instruction language.

    Args:
        expr: Either a string (extension/instruction name) or InstructionPattern
        has_c_ext: Whether C extension is required
        width: Instruction width (32 or 16)

    Returns:
        BDD representing all instructions matching the expression
    """
    bdd_mgr = get_bdd_manager(width)

    if isinstance(expr, str):
        # Extension name or bare instruction name
        instructions = get_extension_instructions(expr)

        if instructions:
            # It's an extension - OR together all instruction BDDs
            result = bdd_mgr.false
            for instr_name in instructions:
                encoding = get_instruction_encoding(instr_name)
                if encoding:
                    instr_width = 16 if encoding.is_compressed else 32
                    if instr_width == width:  # Only include instructions of matching width
                        instr_bdd = pattern_to_bdd(encoding.base_pattern, encoding.base_mask, instr_width)
                        result |= instr_bdd
            return result
        else:
            # Try as instruction name
            encoding = get_instruction_encoding(expr)
            if encoding:
                instr_width = 16 if encoding.is_compressed else 32
                if instr_width == width:
                    return pattern_to_bdd(encoding.base_pattern, encoding.base_mask, instr_width)
                else:
                    return bdd_mgr.false
            else:
                print(f"Warning: Unknown extension or instruction '{expr}'")
                return bdd_mgr.false

    elif isinstance(expr, InstructionPattern):
        # Instruction with constraints - use BDD expansion
        return expand_instruction_pattern_v2_bdd(expr, use_bdd=True)
    else:
        print(f"Warning: Invalid expression type: {type(expr)}")
        return bdd_mgr.false


def expand_expr_v2(expr, has_c_ext: bool = False) -> Set[Tuple[int, int, str, bool]]:
    """
    Expand a v2 expression to a set of instruction patterns.

    Args:
        expr: Either a string (extension name) or InstructionPattern
        has_c_ext: Whether C extension is required

    Returns:
        Set of (pattern, mask, description, is_compressed) tuples
    """
    if isinstance(expr, str):
        # Extension name or bare instruction name
        # Check if it's an extension
        instructions = get_extension_instructions(expr)
        if instructions:
            # It's an extension - expand to all instructions
            result = set()
            for instr_name in instructions:
                encoding = get_instruction_encoding(instr_name)
                if encoding:
                    result.add((encoding.base_pattern, encoding.base_mask, instr_name, encoding.is_compressed))
            return result
        else:
            # Try as instruction name
            encoding = get_instruction_encoding(expr)
            if encoding:
                return {(encoding.base_pattern, encoding.base_mask, expr, encoding.is_compressed)}
            else:
                print(f"Warning: Unknown extension or instruction '{expr}'")
                return set()
    elif isinstance(expr, InstructionPattern):
        # Instruction with constraints
        return expand_instruction_pattern_v2(expr)
    else:
        print(f"Warning: Invalid expression type: {type(expr)}")
        return set()


def instruction_rule_to_pattern(rule: InstructionRule, has_c_ext: bool = False) -> List[Tuple[int, int, str, bool]]:
    """
    Convert an InstructionRule to one or more (pattern, mask, description, is_compressed) tuples.

    Returns a list because some rules with wildcards might expand to multiple patterns.
    When has_c_ext is True, auto-expands to include compressed versions of instructions.

    For 'allow' rules (rule.allow=True), generates complementary patterns that forbid
    everything EXCEPT the allowed pattern.
    """
    results = []

    # Get the base encoding for this instruction
    encoding = get_instruction_encoding(rule.name)
    if not encoding:
        print(f"Warning: Unknown instruction '{rule.name}' at line {rule.line}, skipping")
        return []

    # Start with the base pattern and mask
    pattern = encoding.base_pattern
    mask = encoding.base_mask

    # Track if we're constraining a field with a bit pattern (for allow semantics)
    constrained_field = None
    constrained_field_pos = None
    constrained_field_width = None
    constrained_pattern = None
    constrained_mask = None

    # Apply field constraints
    for constraint in rule.constraints:
        field_name = constraint.field_name
        field_value = constraint.field_value

        # Skip data type constraints - these are semantic metadata, not encoding constraints
        # Data type fields: dtype, rd_dtype, rs1_dtype, rs2_dtype, imm_dtype, etc.
        if field_name == 'dtype' or field_name.endswith('_dtype'):
            # These constraints are for optimization/verification, not instruction encoding
            # They will be handled separately by dedicated code generators
            continue

        # Check if this field exists in the instruction format
        if field_name not in encoding.fields:
            print(f"Warning: Field '{field_name}' not valid for {rule.name} at line {rule.line}")
            continue

        field_pos, field_width = encoding.fields[field_name]

        # Handle bit patterns
        if isinstance(field_value, BitPattern):
            # Validate width matches field width
            if field_value.width != field_width:
                print(f"Error at line {rule.line}: Bit pattern width {field_value.width} doesn't match field width {field_width} for {field_name} in {rule.name}")
                continue

            # Convert bit pattern to pattern/mask
            try:
                pattern_val, pattern_mask = field_value.to_pattern_mask()

                # For 'allow' rules, save the constrained field info
                if rule.allow:
                    constrained_field = field_name
                    constrained_field_pos = field_pos
                    constrained_field_width = field_width
                    constrained_pattern = pattern
                    constrained_mask = mask

                # Apply to instruction pattern/mask
                pattern = set_field(pattern, field_pos, field_width, pattern_val)
                mask = mask | (pattern_mask << field_pos)
                continue
            except ValueError as e:
                print(f"Error at line {rule.line}: Invalid bit pattern for {field_name}: {e}")
                continue

        # Handle wildcards - don't add to mask
        if field_value in ('*', 'x', '_'):
            # Wildcard means we don't care about this field
            # Remove this field from the mask (set those bits to 0)
            field_mask = create_field_mask(field_pos, field_width)
            mask = mask & ~field_mask
            continue

        # Handle register names
        if field_name in ('rd', 'rs1', 'rs2'):
            reg_num = parse_register(field_value)
            if reg_num is None:
                # Try to parse as number
                if isinstance(field_value, int):
                    reg_num = field_value
                else:
                    print(f"Warning: Invalid register '{field_value}' at line {rule.line}")
                    continue
            field_value = reg_num

        # Handle numeric values
        if isinstance(field_value, str):
            # Try to parse as hex/binary/decimal
            try:
                if field_value.startswith('0x'):
                    field_value = int(field_value, 16)
                elif field_value.startswith('0b'):
                    field_value = int(field_value, 2)
                else:
                    field_value = int(field_value)
            except ValueError:
                print(f"Warning: Cannot parse field value '{field_value}' at line {rule.line}")
                continue

        # Set the field in the pattern and add to mask
        pattern = set_field(pattern, field_pos, field_width, field_value)
        mask = mask | create_field_mask(field_pos, field_width)

    # Create description
    desc = f"{rule.name}"
    if rule.constraints:
        constraint_strs = [f"{c.field_name}={c.field_value}" for c in rule.constraints]
        desc += " { " + ", ".join(constraint_strs) + " }"

    # Add the base instruction pattern
    results.append((pattern, mask, desc, encoding.is_compressed))

    # If C extension is required and this is not already a compressed instruction,
    # check if a compressed version exists and add it too
    if has_c_ext and not encoding.is_compressed and not rule.name.startswith('C.'):
        compressed_name = f"C.{rule.name}"
        compressed_encoding = get_instruction_encoding(compressed_name)
        if compressed_encoding:
            # For compressed version, we don't apply field constraints since they're different formats
            # Just outlaw the compressed version entirely
            compressed_desc = f"{compressed_name} (auto-expanded from {rule.name})"
            results.append((compressed_encoding.base_pattern, compressed_encoding.base_mask,
                            compressed_desc, True))

    return results


def generate_sv_module(patterns: List[Tuple[int, int, str]], module_name: str = "instr_outlawed_checker",
                       dtype_assertions: str = "") -> str:
    """Generate SystemVerilog module with pattern-matching assertions."""

    # Determine if we need operand signals for dtype checks
    needs_operands = len(dtype_assertions) > 0

    sv_code = f"""// Auto-generated instruction pattern checker module
// This module checks that certain instruction patterns are never present in the pipeline

module {module_name} (
  input logic        clk_i,
  input logic        rst_ni,
  input logic        instr_valid_i,
  input logic [31:0] instr_rdata_i,
  input logic        instr_is_compressed_i"""

    if needs_operands:
        sv_code += """,
  // Operand signals for data type constraints
  input logic [31:0] alu_operand_a_ex_i,
  input logic [31:0] alu_operand_b_ex_i,
  input logic [31:0] multdiv_operand_a_ex_i,
  input logic [31:0] multdiv_operand_b_ex_i"""

    sv_code += "\n);\n\n"

    # Group by width (assume all 32-bit for now, can add 16-bit later)
    # Handle both 3-tuple and 4-tuple formats (with is_compressed flag)
    patterns_32 = []
    for item in patterns:
        if len(item) == 4:
            p, m, d, _ = item
        else:
            p, m, d = item
        if m <= 0xFFFFFFFF:
            patterns_32.append((p, m, d))

    if patterns_32:
        sv_code += "  // 32-bit outlawed instruction patterns\n"
        sv_code += "  // Using combinational 'assume' so ABC can use them as don't-care conditions\n"
        sv_code += "  // This allows ABC to optimize away logic for these instructions\n"
        for i, (pattern, mask, desc) in enumerate(patterns_32):
            sv_code += f"  // {desc}\n"
            sv_code += f"  // Pattern: 0x{pattern:08x}, Mask: 0x{mask:08x}\n"
            sv_code += f"  // Combinational assumption: when valid, this pattern doesn't occur\n"
            sv_code += f"  always_comb begin\n"
            sv_code += f"    if (rst_ni && instr_valid_i && !instr_is_compressed_i) begin\n"
            sv_code += f"      assume ((instr_rdata_i & 32'h{mask:08x}) != 32'h{pattern:08x});\n"
            sv_code += f"    end\n"
            sv_code += f"  end\n\n"

    if not patterns_32:
        sv_code += "  // No outlawed instruction patterns specified\n\n"

    # Add data type assertions if present
    if dtype_assertions:
        sv_code += dtype_assertions

    sv_code += "endmodule\n"

    return sv_code


def generate_bind_file(module_name: str) -> str:
    """Generate a bind file for the checker module."""
    bind_code = f"""// Auto-generated bind file for {module_name}
// This file binds the instruction checker to the Ibex ID stage

bind ibex_core.id_stage_i {module_name} checker_inst (
  .clk_i                  (clk_i),
  .rst_ni                 (rst_ni),
  .instr_valid_i          (instr_valid_i),
  .instr_rdata_i          (instr_rdata_i),
  .instr_is_compressed_i  (instr_is_compressed_i)
);
"""
    return bind_code


def generate_dtype_assertions(rules: List[InstructionRule], config: Optional[CoreConfig] = None) -> str:
    """
    Generate SystemVerilog assertions for data type constraints.

    Args:
        rules: List of instruction rules
        config: Core configuration (defaults to Ibex if not provided)

    Returns:
        SystemVerilog code to be added to the checker module.
    """
    if config is None:
        config = CoreConfig.default_ibex()

    code = ""
    data_width = config.data_width

    # Collect all InstructionRules with dtype constraints
    dtype_rules = []
    for rule in rules:
        if not isinstance(rule, InstructionRule):
            continue
        has_dtype = any(c.field_name == 'dtype' or c.field_name.endswith('_dtype')
                        for c in rule.constraints)
        if has_dtype:
            dtype_rules.append(rule)

    if not dtype_rules:
        return ""

    code += "  // ========================================\n"
    code += "  // Data type constraint assertions\n"
    code += "  // ========================================\n\n"

    # For each rule with dtype constraints
    for rule in dtype_rules:
        encoding = get_instruction_encoding(rule.name)
        if not encoding:
            continue

        # Extract dtype constraints
        for constraint in rule.constraints:
            field_name = constraint.field_name
            field_value = constraint.field_value

            if not isinstance(field_value, DataTypeSet):
                continue

            dtype_set = field_value

            # Determine which operands to check using config
            mul_div_instrs = {'MUL', 'MULH', 'MULHSU',
                              'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'}
            is_multdiv = rule.name.upper() in mul_div_instrs

            if field_name == 'dtype':
                # Apply to all operands
                operand_fields = []
                if 'rs1' in encoding.fields:
                    signal = config.signals.multdiv_rs1 if is_multdiv else config.signals.alu_rs1
                    operand_fields.append(('rs1', signal))
                if 'rs2' in encoding.fields:
                    signal = config.signals.multdiv_rs2 if is_multdiv else config.signals.alu_rs2
                    operand_fields.append(('rs2', signal))
            elif field_name == 'rs1_dtype':
                signal = config.signals.multdiv_rs1 if is_multdiv else config.signals.alu_rs1
                operand_fields = [('rs1', signal)]
            elif field_name == 'rs2_dtype':
                signal = config.signals.multdiv_rs2 if is_multdiv else config.signals.alu_rs2
                operand_fields = [('rs2', signal)]
            elif field_name == 'rd_dtype':
                # Skip rd for now - would need writeback stage signals
                continue
            else:
                continue

            # Generate check for each operand
            for operand_name, signal_name in operand_fields:
                check_expr = generate_dtype_set_check_expr(
                    dtype_set, signal_name, data_width)

                # Create instruction match condition using config signal names
                instr_match = f"(({config.signals.instruction_data} & {data_width}'h{encoding.base_mask:08x}) == {data_width}'h{encoding.base_pattern:08x})"

                # Determine assertion condition based on negation
                if dtype_set.negated:
                    # Negated: ALLOW only these types (forbid all others)
                    # Assert that operand MUST match one of the types
                    condition = check_expr
                    semantic = "allow only"
                else:
                    # Not negated: FORBID these types
                    # Assert that operand must NOT match any of the types
                    condition = f"!{check_expr}"
                    semantic = "forbid"

                code += f"  // {rule.name} {operand_name}: {semantic} {dtype_set}\n"
                code += f"  // Gated by reset to allow ABC scorr (init state must satisfy constraints)\n"
                code += f"  always_comb begin\n"
                code += f"    if ({instr_match}) begin\n"
                code += f"      assume (!rst_ni || ({condition}));\n"
                code += f"    end\n"
                code += f"  end\n\n"

    return code


def generate_module_dtype_assertions(rules: List[InstructionRule],
                                      module_config: ModuleConfig,
                                      data_width: int = 32) -> str:
    """
    Generate SystemVerilog data type assertions for a specific module.

    This is used for hierarchical synthesis where a module (like ibex_ex_block)
    is synthesized separately. The operand signals are PRIMARY INPUTS to the
    module, so ABC can optimize with these constraints.

    Args:
        rules: List of instruction rules with dtype constraints
        module_config: Module configuration with signal names
        data_width: Data width (32 for RV32, 64 for RV64)

    Returns:
        SystemVerilog assumption code for the module
    """
    if not module_config.signals:
        raise ValueError(f"Module '{module_config.name}' has no signal configuration")

    signals = module_config.signals
    code = ""

    # Collect all InstructionRules with dtype constraints
    dtype_rules = []
    for rule in rules:
        if not isinstance(rule, InstructionRule):
            continue
        has_dtype = any(c.field_name == 'dtype' or c.field_name.endswith('_dtype')
                        for c in rule.constraints)
        if has_dtype:
            dtype_rules.append(rule)

    if not dtype_rules:
        return ""

    code += "  // ========================================\n"
    code += f"  // Data type constraints for {module_config.name}\n"
    code += "  // Operands are PRIMARY INPUTS - satisfiable at init state\n"
    code += "  // ========================================\n\n"

    # For each rule with dtype constraints
    for rule in dtype_rules:
        encoding = get_instruction_encoding(rule.name)
        if not encoding:
            continue

        # Extract dtype constraints
        for constraint in rule.constraints:
            field_name = constraint.field_name
            field_value = constraint.field_value

            if not isinstance(field_value, DataTypeSet):
                continue

            dtype_set = field_value

            # Determine which operands to check using module config
            mul_div_instrs = {'MUL', 'MULH', 'MULHSU',
                              'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'}
            is_multdiv = rule.name.upper() in mul_div_instrs

            if field_name == 'dtype':
                # Apply to all operands
                operand_fields = []
                if 'rs1' in encoding.fields:
                    signal = signals.multdiv_rs1 if is_multdiv else signals.alu_rs1
                    operand_fields.append(('rs1', signal))
                if 'rs2' in encoding.fields:
                    signal = signals.multdiv_rs2 if is_multdiv else signals.alu_rs2
                    operand_fields.append(('rs2', signal))
            elif field_name == 'rs1_dtype':
                signal = signals.multdiv_rs1 if is_multdiv else signals.alu_rs1
                operand_fields = [('rs1', signal)]
            elif field_name == 'rs2_dtype':
                signal = signals.multdiv_rs2 if is_multdiv else signals.alu_rs2
                operand_fields = [('rs2', signal)]
            elif field_name == 'rd_dtype':
                # Skip rd for now - would need writeback stage signals
                continue
            else:
                continue

            # Generate check for each operand
            for operand_name, signal_name in operand_fields:
                check_expr = generate_dtype_set_check_expr(
                    dtype_set, signal_name, data_width)

                # Create instruction match condition using module signal names
                instr_match = f"(({signals.instruction_data} & {data_width}'h{encoding.base_mask:08x}) == {data_width}'h{encoding.base_pattern:08x})"

                # Determine assertion condition based on negation
                if dtype_set.negated:
                    # Negated: ALLOW only these types (forbid all others)
                    # Assert that operand MUST match one of the types
                    condition = check_expr
                    semantic = "allow only"
                else:
                    # Not negated: FORBID these types
                    # Assert that operand must NOT match any of the types
                    condition = f"!{check_expr}"
                    semantic = "forbid"

                code += f"  // {rule.name} {operand_name}: {semantic} {dtype_set}\n"
                code += f"  // Gated by reset - operands are PRIMARY INPUTS (no init state issue)\n"
                code += f"  always_comb begin\n"
                code += f"    if ({instr_match}) begin\n"
                code += f"      assume (!rst_ni || ({condition}));\n"
                code += f"    end\n"
                code += f"  end\n\n"

    return code


def generate_timing_constraints(instr_hit_latency, instr_miss_latency,
                                data_hit_latency, data_miss_latency,
                                locality_bits) -> str:
    """
    Generate SystemVerilog timing constraints for cache-aware optimization.

    Args:
        instr_hit_latency: Max cycles for instruction cache hit (-1 to disable)
        instr_miss_latency: Max cycles for instruction cache miss (-1 to disable)
        data_hit_latency: Max cycles for data cache hit (-1 to disable)
        data_miss_latency: Max cycles for data cache miss (-1 to disable)
        locality_bits: Number of address high bits for locality detection (unused)

    Returns:
        SystemVerilog code with timing constraints
    """
    # Check if we have any timing constraints to generate
    has_instr_constraints = instr_hit_latency != -1 or instr_miss_latency != -1
    has_data_constraints = data_hit_latency != -1 or data_miss_latency != -1

    if not has_instr_constraints and not has_data_constraints:
        return ""  # No timing constraints to generate

    code = """
  // ========================================
  // Cache-Aware Timing Constraints
  // ========================================
  // Models realistic cache hit/miss behavior for ABC optimization
  // Injected at ibex_core level for full signal visibility
"""

    # Generate instruction cache constraints if needed
    if has_instr_constraints:
        code += """
  // Instruction cache timing tracking
  logic [2:0] instr_stall_counter_q;
  logic instr_likely_miss;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      instr_stall_counter_q <= 3'b0;
      instr_likely_miss <= 1'b0;
    end else begin
      if (instr_req_o && !instr_gnt_i) begin
        instr_stall_counter_q <= instr_stall_counter_q + 1;
        // After 1 cycle of stalling, assume cache miss
        instr_likely_miss <= (instr_stall_counter_q >= 1);
      end else begin
        instr_stall_counter_q <= 3'b0;
        instr_likely_miss <= 1'b0;
      end

    end
  end
"""

    # Generate data cache constraints if needed
    if has_data_constraints:
        code += """
  // Data cache timing tracking
  logic [2:0] data_stall_counter_q;
  logic data_likely_miss;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      data_stall_counter_q <= 3'b0;
      data_likely_miss <= 1'b0;
    end else begin
      if (data_req_out && !data_gnt_i) begin
        data_stall_counter_q <= data_stall_counter_q + 1;
        data_likely_miss <= (data_stall_counter_q >= 1);
      end else if (data_rvalid_i) begin
        data_stall_counter_q <= 3'b0;
        data_likely_miss <= 1'b0;
      end

    end
  end
"""

    # Generate timing assumptions
    code += """
  // Cache-aware timing assumptions
  always_comb begin
    if (rst_ni) begin"""

    if has_instr_constraints:
        code += """
      // Instruction cache: different bounds for hit vs miss"""
        if instr_hit_latency != -1 and instr_miss_latency != -1:
            code += f"""
      if (!instr_likely_miss) begin
        assume(instr_stall_counter_q <= 3'd{instr_hit_latency});  // Cache hit: {instr_hit_latency} cycle max
      end else begin
        assume(instr_stall_counter_q <= 3'd{instr_miss_latency});  // Cache miss: up to {instr_miss_latency} cycles
      end"""
        elif instr_hit_latency != -1:
            code += f"""
      if (!instr_likely_miss) begin
        assume(instr_stall_counter_q <= 3'd{instr_hit_latency});  // Cache hit: {instr_hit_latency} cycle max
      end"""
        elif instr_miss_latency != -1:
            code += f"""
      if (instr_likely_miss) begin
        assume(instr_stall_counter_q <= 3'd{instr_miss_latency});  // Cache miss: up to {instr_miss_latency} cycles
      end"""

        if instr_miss_latency != -1:
            code += f"""

      // Force completion at maximum latency
      if (instr_stall_counter_q == 3'd{instr_miss_latency}) begin
        assume(instr_gnt_i);  // Must grant at max stall
      end"""

    if has_data_constraints:
        code += """
      // Data cache: different bounds for hit vs miss"""
        if data_hit_latency != -1 and data_miss_latency != -1:
            code += f"""
      if (!data_likely_miss) begin
        assume(data_stall_counter_q <= 3'd{data_hit_latency});   // Cache hit: {data_hit_latency} cycle max
      end else begin
        assume(data_stall_counter_q <= 3'd{data_miss_latency});   // Cache miss: up to {data_miss_latency} cycles
      end"""
        elif data_hit_latency != -1:
            code += f"""
      if (!data_likely_miss) begin
        assume(data_stall_counter_q <= 3'd{data_hit_latency});   // Cache hit: {data_hit_latency} cycle max
      end"""
        elif data_miss_latency != -1:
            code += f"""
      if (data_likely_miss) begin
        assume(data_stall_counter_q <= 3'd{data_miss_latency});   // Cache miss: up to {data_miss_latency} cycles
      end"""

        if data_miss_latency != -1:
            code += f"""

      // Force completion at maximum latency
      if (data_stall_counter_q == 3'd{data_miss_latency}) begin
        assume(data_gnt_i);  // Must grant at max stall
      end"""

    code += """
    end
  end
"""

    return code


def generate_inline_assumptions_bdd(sigma_bdd, register_constraint=None, pc_bits=None, config=None):
    """
    Generate SystemVerilog assumptions from a BDD (v2 with BDD mode).

    Args:
        sigma_bdd: BDD expression representing allowed instructions
        register_constraint: Register range constraint if specified
        pc_bits: Number of PC address bits
        config: Core configuration

    Returns:
        SystemVerilog code string
    """
    if config is None:
        from .config import CoreConfig
        config = CoreConfig.default_ibex()

    from .bdd_ops import bdd_to_systemverilog_function

    instr_data = config.signals.instruction_data
    has_compressed_check = config.signals.has_compressed_check

    code = "\n  // ========================================\n"
    code += "  // Auto-generated instruction constraints\n"
    code += f"  // Target core: {config.core_name} ({config.architecture})\n"
    code += f"  // Inject into: {config.injection.module_path}\n"
    code += f"  // Location: {config.injection.description}\n"
    code += "  // Generated from BDD (Binary Decision Diagram)\n"
    code += "  // ========================================\n\n"

    # Add PC constraint if specified
    if pc_bits is not None:
        addr_space_kb = (2 ** pc_bits) // 1024
        code += f"  // PC address space constraint: {pc_bits}-bit address space ({addr_space_kb}KB)\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || {config.signals.pc}[31:{pc_bits}] == {32-pc_bits}'b0);\n"
        code += "  end\n\n"

    # Generate function from BDD
    code += "  // BDD-based constraint function\n"
    func_code = bdd_to_systemverilog_function(sigma_bdd, "matches_allowed_instruction", "instr", 32)
    code += func_code
    code += "\n"

    # Use function in assumption
    code += "  // Instruction must match allowed set\n"
    code += "  always_comb begin\n"
    if has_compressed_check:
        code += f"    assume (!rst_ni || ({instr_data}[1:0] != 2'b11) || matches_allowed_instruction({instr_data}));\n"
    else:
        code += f"    assume (!rst_ni || matches_allowed_instruction({instr_data}));\n"
    code += "  end\n\n"

    # TODO: Add register constraints if specified

    return code


def generate_inline_assumptions(patterns, required_extensions: Set[str] = None,
                                register_constraint: Optional[RegisterConstraintRule] = None,
                                pc_bits: Optional[int] = None,
                                config: Optional[CoreConfig] = None,
                                instruction_rules: Optional[List[InstructionRule]] = None,
                                v2_mode: bool = False):
    """Generate assumptions to inject directly into ID stage (no separate module).

    Args:
        patterns: List of (pattern, mask, description, is_compressed) tuples
        required_extensions: Set of required RISC-V extensions (v1 only)
        register_constraint: Register range constraint if specified
        pc_bits: Number of PC address bits (e.g., 16 for 64KB). If provided, constrains PC[31:pc_bits] to 0
        config: Core configuration (defaults to Ibex if not provided)
        instruction_rules: List of instruction rules (for data type constraint generation)
        v2_mode: If True, generate positive assertions (allow ONLY these patterns) for DSL v2
    """
    if config is None:
        config = CoreConfig.default_ibex()

    data_width = config.data_width
    instr_data = config.signals.instruction_data
    has_compressed_check = config.signals.has_compressed_check

    code = "\n  // ========================================\n"
    code += "  // Auto-generated instruction constraints\n"
    code += f"  // Target core: {config.core_name} ({config.architecture})\n"
    code += f"  // Inject into: {config.injection.module_path}\n"
    code += f"  // Location: {config.injection.description}\n"
    code += "  // ========================================\n\n"

    # Add PC constraint if specified
    if pc_bits is not None:
        addr_space_kb = (2 ** pc_bits) // 1024
        code += f"  // PC address space constraint: {pc_bits}-bit address space ({addr_space_kb}KB)\n"
        code += f"  // Unconditional assumption for ABC optimization\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || {config.signals.pc}[{data_width-1}:{pc_bits}] == {data_width-pc_bits}'b0);\n"
        code += "  end\n\n"

    # Note: No compression bit consistency check needed
    # We directly check instr_rdata_i[1:0] in each assumption instead of using instr_is_compressed_i
    # This reduces signal dependencies and gives ABC more optimization freedom

    # Generate register constraints
    if register_constraint:
        min_reg = register_constraint.min_reg
        max_reg = register_constraint.max_reg
        code += f"  // Register constraint: only x{min_reg}-x{max_reg} allowed ({max_reg - min_reg + 1} registers)\n"
        code += "  // Constraints applied based on instruction format (not all instructions use all fields)\n\n"

        # RISC-V instruction formats and which register fields they use:
        # Opcode is bits [6:0], we use this to determine format

        # R-type (opcode[6:2] = 01100, 01110, 10100, 10110): rd, rs1, rs2
        code += "  // R-type instructions (OP, OP-32): rd, rs1, rs2\n"
        code += f"  wire is_r_type = ({instr_data}[6:2] == 5'b01100) ||  // OP (ADD, SUB, etc.)\n"
        code += f"                   ({instr_data}[6:2] == 5'b01110) ||  // OP-32 (ADDW, SUBW, etc.)\n"
        code += f"                   ({instr_data}[6:2] == 5'b10100) ||  // OP-FP\n"
        code += f"                   ({instr_data}[6:2] == 5'b10110);    // OP-V\n"
        code += "  always_comb begin\n"
        if has_compressed_check:
            code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_r_type ||\n"
        else:
            code += f"    assume (!is_r_type ||\n"
        code += f"            (({instr_data}[11:7] <= 5'd{max_reg}) &&   // rd\n"
        code += f"             ({instr_data}[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             ({instr_data}[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # I-type (loads, JALR, OP-IMM): rd, rs1
        code += "  // I-type instructions (LOAD, OP-IMM, JALR): rd, rs1\n"
        code += f"  wire is_i_type = ({instr_data}[6:2] == 5'b00000) ||  // LOAD\n"
        code += f"                   ({instr_data}[6:2] == 5'b00100) ||  // OP-IMM\n"
        code += f"                   ({instr_data}[6:2] == 5'b00110) ||  // OP-IMM-32\n"
        code += f"                   ({instr_data}[6:2] == 5'b11001);    // JALR\n"
        code += "  always_comb begin\n"

        if has_compressed_check:
            code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_i_type ||\n"
        else:
            code += f"    assume (!is_i_type ||\n"
        code += f"            (({instr_data}[11:7] <= 5'd{max_reg}) &&   // rd\n"
        code += f"             ({instr_data}[19:15] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # S-type (stores): rs1, rs2 (no rd - bits [11:7] are immediate)
        code += "  // S-type instructions (STORE): rs1, rs2 (no rd)\n"
        code += f"  wire is_s_type = ({instr_data}[6:2] == 5'b01000);    // STORE\n"
        code += "  always_comb begin\n"

        if has_compressed_check:
            code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_s_type ||\n"
        else:
            code += f"    assume (!is_s_type ||\n"
        code += f"            (({instr_data}[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             ({instr_data}[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # B-type (branches): rs1, rs2 (no rd)
        code += "  // B-type instructions (BRANCH): rs1, rs2 (no rd)\n"
        code += f"  wire is_b_type = ({instr_data}[6:2] == 5'b11000);    // BRANCH\n"
        code += "  always_comb begin\n"
        if has_compressed_check:
            code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_b_type ||\n"
        else:
            code += f"    assume (!is_b_type ||\n"
        code += f"            (({instr_data}[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             ({instr_data}[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # U-type (LUI, AUIPC): rd only
        code += "  // U-type instructions (LUI, AUIPC): rd only\n"
        code += f"  wire is_u_type = ({instr_data}[6:2] == 5'b01101) ||  // LUI\n"
        code += f"                   ({instr_data}[6:2] == 5'b00101);    // AUIPC\n"
        code += "  always_comb begin\n"

        if has_compressed_check:
            code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_u_type ||\n"
        else:
            code += f"    assume (!is_u_type ||\n"
        code += f"            ({instr_data}[11:7] <= 5'd{max_reg}));\n"
        code += "  end\n\n"

        # J-type (JAL): rd only
        code += "  // J-type instructions (JAL): rd only\n"
        code += f"  wire is_j_type = ({instr_data}[6:2] == 5'b11011);    // JAL\n"
        code += "  always_comb begin\n"

        if has_compressed_check:
            code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_j_type ||\n"
        else:
            code += f"    assume (!is_j_type ||\n"
        code += f"            ({instr_data}[11:7] <= 5'd{max_reg}));\n"
        code += "  end\n\n"

    # Generate data type constraints
    if instruction_rules:
        dtype_code = generate_dtype_assertions(instruction_rules, config)
        if dtype_code:
            code += dtype_code

    # V2: Generate positive assertions (allow ONLY these patterns)
    if v2_mode:
        if not patterns:
            code += "  // V2: No instructions allowed (empty instruction set)\n\n"
            return code

        # Separate patterns by compression
        patterns_32bit = []
        patterns_16bit = []
        for item in patterns:
            if len(item) == 4:
                pattern, mask, desc, is_compressed = item
            else:
                pattern, mask, desc = item
                is_compressed = False

            if is_compressed:
                patterns_16bit.append((pattern, mask, desc))
            else:
                patterns_32bit.append((pattern, mask, desc))

        code += "  // V2: Positive assertion - instruction must be one of these allowed patterns\n"
        code += f"  // Allowed instruction set: {len(patterns)} patterns\n"

        # Generate 32-bit instruction positive constraint
        if patterns_32bit:
            code += "  // 32-bit instructions: allow ONLY these patterns\n"
            code += "  always_comb begin\n"
            if has_compressed_check:
                code += f"    assume (!rst_ni || ({instr_data}[1:0] != 2'b11) || (\n"
            else:
                code += f"    assume (!rst_ni || (\n"

            # Generate OR of all allowed patterns
            for i, (pattern, mask, desc) in enumerate(patterns_32bit):
                is_last = (i == len(patterns_32bit) - 1)
                connector = "" if is_last else " ||"
                code += f"      (({instr_data}[{data_width-1}:0] & {data_width}'h{mask:08x}) == {data_width}'h{pattern:08x}){connector}  // {desc}\n"

            if has_compressed_check:
                code += "    ));\n"
            else:
                code += "    );\n"
            code += "  end\n\n"

        # Generate 16-bit compressed instruction positive constraint
        if patterns_16bit:
            code += "  // 16-bit compressed instructions: allow ONLY these patterns\n"
            code += "  always_comb begin\n"
            code += f"    assume (!rst_ni || ({instr_data}[1:0] == 2'b11) || (\n"

            for i, (pattern, mask, desc) in enumerate(patterns_16bit):
                is_last = (i == len(patterns_16bit) - 1)
                connector = "" if is_last else " ||"
                code += f"      (({instr_data}[15:0] & 16'h{mask:04x}) == 16'h{pattern:04x}){connector}  // {desc}\n"

            code += "    ));\n"
            code += "  end\n\n"

        return code

    # V1: Generate positive constraints from required extensions
    valid_patterns = []  # Define at this scope so it's accessible later
    if required_extensions:
        code += "  // Positive constraint: instruction must be from required extensions\n"
        code += f"  // Required: {', '.join(sorted(required_extensions))}\n"

        # Extract outlawed instruction names from patterns for filtering
        outlawed_names = set()
        for item in patterns:
            if len(item) == 4:  # New format with is_compressed flag
                pattern, mask, desc, is_compressed = item
            else:  # Legacy format
                pattern, mask, desc = item
            # Extract instruction name from description (format: "INSTR" or "INSTR { constraints }")
            instr_name = desc.split()[0].split('{')[0].strip()
            outlawed_names.add(instr_name)

        if outlawed_names:
            code += f"  // Excluding outlawed: {', '.join(sorted(outlawed_names))}\n"

        # Collect all valid instruction patterns from required extensions
        # BUT exclude those that are outlawed
        for ext in required_extensions:
            ext_instrs = get_extension_instructions(ext)
            if ext_instrs:
                for name, encoding in ext_instrs.items():
                    # Skip if this instruction is outlawed
                    if name not in outlawed_names:
                        valid_patterns.append(
                            (encoding.base_pattern, encoding.base_mask, f"{name} ({ext})"))

        if valid_patterns:
            code += "  // Instruction must match one of these valid patterns (OR of all valid instructions)\n"
            code += "  always_comb begin\n"
            if has_compressed_check:
                code += f"    assume (!rst_ni || ({instr_data}[1:0] != 2'b11) || (\n"
            else:
                code += f"    assume (!rst_ni || (\n"

            # Generate OR of all valid instruction patterns
            for i, (pattern, mask, desc) in enumerate(valid_patterns):
                is_last = (i == len(valid_patterns) - 1)
                connector = "" if is_last else " ||"
                code += f"      (({instr_data} & 32'h{mask:08x}) == 32'h{pattern:08x}){connector}  // {desc}\n"

            if has_compressed_check:
                code += "    ));\n"
            else:
                code += "    );\n"
            code += "  end\n\n"

    if not patterns:
        if not required_extensions:
            code += "  // No instruction constraints specified\n\n"
        return code

    # If we have positive constraints from required extensions, we don't need negative constraints
    # The positive constraint already excludes outlawed instructions
    if required_extensions and valid_patterns:
        code += "  // Note: Negative constraints not needed - outlawed instructions already excluded from positive list\n\n"
        return code

    # Only generate negative constraints if we don't have positive constraints
    # (This is for backward compatibility or when no extensions are specified)

    # Separate patterns by compression
    patterns_32bit = []
    patterns_16bit = []
    for item in patterns:
        if len(item) == 4:  # New format with is_compressed flag
            pattern, mask, desc, is_compressed = item
        else:  # Legacy format without is_compressed flag
            pattern, mask, desc = item
            is_compressed = False

        if is_compressed:
            patterns_16bit.append((pattern, mask, desc))
        else:
            patterns_32bit.append((pattern, mask, desc))

    # Generate constraints for 32-bit instructions
    if patterns_32bit:
        code += "  // 32-bit instruction outlawed patterns\n"
        code += "  // When out of reset and instruction is uncompressed, these patterns don't occur\n"
        for pattern, mask, desc in patterns_32bit:
            code += f"  // {desc}: Pattern=0x{pattern:08x}, Mask=0x{mask:08x}\n"
            code += f"  always_comb begin\n"
            code += f"    assume (!rst_ni || ({instr_data}[1:0] != 2'b11) || (({instr_data}[{data_width-1}:0] & {data_width}'h{mask:08x}) != {data_width}'h{pattern:08x}));\n"
            code += f"  end\n\n"

    # Generate constraints for 16-bit compressed instructions
    if patterns_16bit:
        code += "  // 16-bit compressed instruction outlawed patterns\n"
        code += "  // When out of reset and instruction is compressed, these patterns don't occur\n"
        for pattern, mask, desc in patterns_16bit:
            code += f"  // {desc}: Pattern=0x{pattern:04x}, Mask=0x{mask:04x}\n"
            code += f"  always_comb begin\n"
            code += f"    assume (!rst_ni || ({instr_data}[1:0] == 2'b11) || (({instr_data}[15:0] & 16'h{mask:04x}) != 16'h{pattern:04x}));\n"
            code += f"  end\n\n"

    return code


def process_v2_rules_bdd(rules: List, has_c_ext: bool = False, use_bdd: bool = True) -> List[Tuple[int, int, str, bool]]:
    """
    Process v2 rules sequentially using BDD operations.

    V2 Semantics with BDDs:
        σ₀ = ∅ (BDD.false)
        include e: σ := σ ∨ ⟦e⟧  (BDD OR)
        forbid e:  σ := σ ∧ ¬⟦e⟧ (BDD AND NOT)

    Args:
        rules: List of v2 rules (IncludeRule, ForbidRule, etc.)
        has_c_ext: Whether C extension is required
        use_bdd: If True, use BDD operations; if False, use legacy set operations

    Returns:
        List of (pattern, mask, description, is_compressed) tuples
    """
    if not use_bdd:
        # Fall back to legacy implementation
        return list(process_v2_rules(rules, has_c_ext))

    bdd_mgr = get_bdd_manager(32)
    sigma = bdd_mgr.false  # σ₀ = ∅ (empty language)

    print("Processing v2 rules sequentially (using BDDs)...")

    for i, rule in enumerate(rules, 1):
        if isinstance(rule, IncludeRule):
            expr_bdd = expand_expr_v2_bdd(rule.expr, has_c_ext, width=32)
            sigma = sigma | expr_bdd  # σ := σ ∪ ⟦e⟧
            count = bdd_size(sigma, 32)
            print(f"  {i}. include: σ now contains {count} instructions")

        elif isinstance(rule, ForbidRule):
            expr_bdd = expand_expr_v2_bdd(rule.expr, has_c_ext, width=32)
            old_count = bdd_size(sigma, 32)
            sigma = sigma & ~expr_bdd  # σ := σ ∧ ¬⟦e⟧
            new_count = bdd_size(sigma, 32)
            removed = old_count - new_count
            print(f"  {i}. forbid: removed {removed} instructions, σ now contains {new_count}")

    final_count = bdd_size(sigma, 32)
    print(f"Final instruction set: {final_count} instructions")

    # Return the BDD itself instead of converting to patterns
    # The caller will generate SV directly from the BDD
    return sigma  # Return BDD expression


def process_v2_rules(rules: List, has_c_ext: bool = False) -> Set[Tuple[int, int, str, bool]]:
    """
    Process v2 rules sequentially using set operations.

    V2 Semantics:
        σ₀ = ∅ (start with empty set)
        include e: σ := σ ∪ ⟦e⟧
        forbid e:  σ := σ \ ⟦e⟧

    Args:
        rules: List of v2 rules (IncludeRule, ForbidRule, etc.)
        has_c_ext: Whether C extension is required

    Returns:
        Final set of allowed instruction patterns
    """
    sigma = set()  # Start with empty set

    print("Processing v2 rules sequentially...")

    for i, rule in enumerate(rules, 1):
        if isinstance(rule, IncludeRule):
            expanded = expand_expr_v2(rule.expr, has_c_ext)
            sigma = sigma | expanded
            print(f"  {i}. include: Added {len(expanded)} patterns, total now: {len(sigma)}")

        elif isinstance(rule, ForbidRule):
            expanded = expand_expr_v2(rule.expr, has_c_ext)
            removed = sigma & expanded
            sigma = sigma - expanded
            print(f"  {i}. forbid: Removed {len(removed)} patterns, total now: {len(sigma)}")

        # Ignore non-v2 rules (config rules handled separately)

    print(f"Final instruction set: {len(sigma)} patterns")
    return sigma


def main():
    parser = argparse.ArgumentParser(
        description='Generate inline SystemVerilog assumptions from instruction DSL'
    )
    parser.add_argument(
        'input_file', help='DSL file containing instruction rules')
    parser.add_argument(
        'output_file', help='Output SystemVerilog file (inline assumptions)')
    parser.add_argument('--config', type=Path,
                        help='Core configuration YAML file (default: builtin Ibex config)')
    parser.add_argument('--target', choices=['ibex', 'boom', 'rocket'],
                        help='Target core shortcut (ibex, boom, rocket)')

    args = parser.parse_args()

    # Load core configuration
    config = load_config(args.config, args.target)
    print(f"Target core: {config.core_name} ({config.architecture})")

    # Read input file
    with open(args.input_file, 'r') as f:
        dsl_text = f.read()

    # Parse DSL
    print(f"Parsing {args.input_file}...")
    try:
        ast = parse_dsl(dsl_text)
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        sys.exit(1)

    print(f"Found {len(ast.rules)} rules")

    # Check DSL version
    dsl_version = ast.version if ast.version is not None else 1
    print(f"DSL version: {dsl_version}")

    # Check if C extension is required
    has_c_ext = has_c_extension_required(ast.rules)
    if has_c_ext:
        print("C extension detected - will auto-expand compressed instructions")

    # Route to appropriate code generation based on version
    if dsl_version == 2:
        # V2: Sequential evaluation with include/forbid
        print("Using v2 semantics (sequential include/forbid)")

        # Extract config rules (still apply in v2)
        register_constraint = None
        pc_constraint = None
        timing_constraint = None
        instruction_rules = []  # For dtype processing

        for rule in ast.rules:
            if isinstance(rule, RegisterConstraintRule):
                if register_constraint is not None:
                    print(f"Warning: Multiple register constraints found, using the last one (x{rule.min_reg}-x{rule.max_reg})")
                register_constraint = rule
            elif isinstance(rule, PcConstraintRule):
                if pc_constraint is not None:
                    print(f"Warning: Multiple PC constraints found, using the last one ({rule.pc_bits} bits)")
                pc_constraint = rule
            elif isinstance(rule, TimingConstraintRule):
                if timing_constraint is None:
                    timing_constraint = {}
                timing_constraint[rule.param_name] = rule.value

        # Process v2 rules sequentially (using set-based approach for explicit patterns)
        # BDD mode generates compact functions but loses pattern visibility for debugging
        # Use set-based approach to generate explicit pattern matching with detailed comments
        sigma_patterns = process_v2_rules_bdd(ast.rules, has_c_ext, use_bdd=False)

        print(f"Final instruction set: {len(sigma_patterns)} patterns")

        required_extensions = set()  # Not used in v2, but keep for compatibility
        if register_constraint:
            print(f"Register constraint: x{register_constraint.min_reg}-x{register_constraint.max_reg}")
        if pc_constraint:
            addr_space_kb = (2 ** pc_constraint.pc_bits) // 1024
            print(f"PC constraint: {pc_constraint.pc_bits} bits ({addr_space_kb}KB address space)")
        if timing_constraint:
            timing_parts = [f"{p}={v}" for p, v in timing_constraint.items()]
            print(f"Timing constraints: {', '.join(timing_parts)}")

        print(f"Generating v2 inline assumptions with explicit patterns...")
        pc_bits = pc_constraint.pc_bits if pc_constraint else None
        code = generate_inline_assumptions(
            list(sigma_patterns), required_extensions, register_constraint, pc_bits,
            config, instruction_rules, v2_mode=True)

        with open(args.output_file, 'w') as f:
            f.write(code)
        print(f"Successfully wrote inline assumptions to {args.output_file}")

        if timing_constraint:
            print(f"Generating cache-aware timing constraints...")
            timing_code = generate_timing_constraints(
                instr_hit_latency=timing_constraint.get('instr_hit_latency', -1),
                instr_miss_latency=timing_constraint.get('instr_miss_latency', -1),
                data_hit_latency=timing_constraint.get('data_hit_latency', -1),
                data_miss_latency=timing_constraint.get('data_miss_latency', -1),
                locality_bits=timing_constraint.get('locality_bits', -1)
            )
            timing_output = args.output_file[:-3] + '_timing.sv' if args.output_file.endswith('.sv') else args.output_file + '_timing.sv'
            with open(timing_output, 'w') as f:
                f.write(timing_code)
            print(f"Successfully wrote timing constraints to {timing_output}")

        return  # Exit early for v2

    # V1: Original outlawing semantics (rest of function unchanged)
    print("Using v1 semantics (require + instruction outlawing)")

    # Separate rules into required extensions, register constraints, PC constraints, and outlawed patterns, timing constraints
    required_extensions = set()
    register_constraint = None
    timing_constraint = None
    pc_constraint = None
    patterns = []
    instruction_rules = []  # Keep instruction rules for dtype processing

    for rule in ast.rules:
        if isinstance(rule, RequireRule):
            required_extensions.add(rule.extension)
        elif isinstance(rule, RegisterConstraintRule):
            if register_constraint is not None:
                print(f"Warning: Multiple register constraints found, using the last one (x{rule.min_reg}-x{rule.max_reg})")
            register_constraint = rule
        elif isinstance(rule, TimingConstraintRule):
            # Collect timing constraints - each rule sets one parameter
            if timing_constraint is None:
                timing_constraint = {}

            # Set the specific parameter from this rule
            timing_constraint[rule.param_name] = rule.value
        elif isinstance(rule, PcConstraintRule):
            if pc_constraint is not None:
                print(f"Warning: Multiple PC constraints found, using the last one ({rule.pc_bits} bits)")
            pc_constraint = rule
        elif isinstance(rule, InstructionRule):
            instruction_rules.append(rule)  # Save for dtype processing

            if rule.allow:
                # TODO: Implement proper 'allow' semantics
                # 'allow' rules define patterns that ARE allowed
                # We need to generate patterns that forbid everything EXCEPT the allowed pattern
                # For now, print a warning
                print(f"Warning: 'allow' keyword at line {rule.line} is not fully implemented yet")
                print(f"  The 'allow' rule will be treated as a regular outlawing rule")
                print(f"  Full implementation requires generating complementary patterns")

            rule_patterns = instruction_rule_to_pattern(rule, has_c_ext)
            patterns.extend(rule_patterns)
        elif isinstance(rule, PatternRule):
            desc = rule.description if rule.description else f"Pattern at line {rule.line}"
            # PatternRules are always 32-bit
            patterns.append((rule.pattern, rule.mask, desc, False))

    if required_extensions:
        print(f"Required extensions: {', '.join(sorted(required_extensions))}")
    if register_constraint:
        print(f"Register constraint: x{register_constraint.min_reg}-x{register_constraint.max_reg} ({register_constraint.max_reg - register_constraint.min_reg + 1} registers)")

    if timing_constraint:
        timing_parts = []
        for param, value in timing_constraint.items():
            timing_parts.append(f"{param}={value}")
        print(f"Timing constraints: {', '.join(timing_parts)}")
    print(f"Generated {len(patterns)} outlawed patterns")

    if pc_constraint:
        addr_space_kb = (2 ** pc_constraint.pc_bits) // 1024
        print(f"PC constraint: {pc_constraint.pc_bits} bits ({addr_space_kb}KB address space)")
    print(f"Generated {len(patterns)} outlawed patterns")

    # Generate inline assumptions code
    pc_bits = pc_constraint.pc_bits if pc_constraint else None

    print("Generating inline SystemVerilog assumptions...")
    code = generate_inline_assumptions(
        patterns, required_extensions, register_constraint, pc_bits, config, instruction_rules)

    with open(args.output_file, 'w') as f:
        f.write(code)
    print(f"Successfully wrote inline assumptions to {args.output_file}")
    # Generate separate timing constraints file if timing constraint is specified
    if timing_constraint:
        print(f"Generating cache-aware timing constraints...")
        timing_code = generate_timing_constraints(
            instr_hit_latency=timing_constraint.get('instr_hit_latency', -1),
            instr_miss_latency=timing_constraint.get('instr_miss_latency', -1),
            data_hit_latency=timing_constraint.get('data_hit_latency', -1),
            data_miss_latency=timing_constraint.get('data_miss_latency', -1),
            locality_bits=timing_constraint.get('locality_bits', -1)
        )

        # Determine timing output file
        if args.output_file.endswith('.sv'):
            timing_output = args.output_file[:-3] + '_timing.sv'
        else:
            timing_output = args.output_file + '_timing.sv'

        with open(timing_output, 'w') as f:
            f.write(timing_code)
        print(f"Successfully wrote timing constraints to {timing_output}")


if __name__ == '__main__':
    main()
