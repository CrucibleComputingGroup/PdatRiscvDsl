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
from .parser import parse_dsl, InstructionRule, PatternRule, FieldConstraint, RequireRule, RegisterConstraintRule, PcConstraintRule, DataTypeSet
from .encodings import (
    get_instruction_encoding, parse_register, set_field, create_field_mask,
    get_extension_instructions
)
from .dtype_codegen import generate_dtype_check_expr, generate_dtype_set_check_expr
from .config import CoreConfig, load_config

def has_c_extension_required(rules: List) -> bool:
    """Check if RV32C or RV64C extension is required in the rules."""
    for rule in rules:
        if isinstance(rule, RequireRule):
            if rule.extension.upper() in ('RV32C', 'RV64C'):
                return True
    return False

def instruction_rule_to_pattern(rule: InstructionRule, has_c_ext: bool = False) -> List[Tuple[int, int, str, bool]]:
    """
    Convert an InstructionRule to one or more (pattern, mask, description, is_compressed) tuples.

    Returns a list because some rules with wildcards might expand to multiple patterns.
    When has_c_ext is True, auto-expands to include compressed versions of instructions.
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

    # Collect all rules with dtype constraints
    dtype_rules = []
    for rule in rules:
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
            mul_div_instrs = {'MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'}
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
                check_expr = generate_dtype_set_check_expr(dtype_set, signal_name, data_width)

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
                code += f"  // Unconditional assumption (no rst_ni/instr_valid_i check)\n"
                code += f"  // This allows ABC to use dtype constraint for optimization\n"
                code += f"  always_comb begin\n"
                code += f"    if ({instr_match}) begin\n"
                code += f"      assume ({condition});\n"
                code += f"    end\n"
                code += f"  end\n\n"

    return code

def generate_inline_assumptions(patterns, required_extensions: Set[str] = None,
                               register_constraint: Optional[RegisterConstraintRule] = None,
                               pc_bits: Optional[int] = None,
                               config: Optional[CoreConfig] = None):
    """Generate assumptions to inject directly into ID stage (no separate module).

    Args:
        patterns: List of (pattern, mask, description, is_compressed) tuples
        required_extensions: Set of required RISC-V extensions
        register_constraint: Register range constraint if specified
        pc_bits: Number of PC address bits (e.g., 16 for 64KB). If provided, constrains PC[31:pc_bits] to 0
        config: Core configuration (defaults to Ibex if not provided)
    """
    if config is None:
        config = CoreConfig.default_ibex()

    data_width = config.data_width
    instr_data = config.signals.instruction_data

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
        code += f"    assume ({config.signals.pc}[{data_width-1}:{pc_bits}] == {data_width-pc_bits}'b0);\n"
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
        code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_r_type ||\n"
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
        code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_i_type ||\n"
        code += f"            (({instr_data}[11:7] <= 5'd{max_reg}) &&   // rd\n"
        code += f"             ({instr_data}[19:15] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # S-type (stores): rs1, rs2 (no rd - bits [11:7] are immediate)
        code += "  // S-type instructions (STORE): rs1, rs2 (no rd)\n"
        code += f"  wire is_s_type = ({instr_data}[6:2] == 5'b01000);    // STORE\n"
        code += "  always_comb begin\n"
        code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_s_type ||\n"
        code += f"            (({instr_data}[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             ({instr_data}[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # B-type (branches): rs1, rs2 (no rd)
        code += "  // B-type instructions (BRANCH): rs1, rs2 (no rd)\n"
        code += f"  wire is_b_type = ({instr_data}[6:2] == 5'b11000);    // BRANCH\n"
        code += "  always_comb begin\n"
        code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_b_type ||\n"
        code += f"            (({instr_data}[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             ({instr_data}[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # U-type (LUI, AUIPC): rd only
        code += "  // U-type instructions (LUI, AUIPC): rd only\n"
        code += f"  wire is_u_type = ({instr_data}[6:2] == 5'b01101) ||  // LUI\n"
        code += f"                   ({instr_data}[6:2] == 5'b00101);    // AUIPC\n"
        code += "  always_comb begin\n"
        code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_u_type ||\n"
        code += f"            ({instr_data}[11:7] <= 5'd{max_reg}));\n"
        code += "  end\n\n"

        # J-type (JAL): rd only
        code += "  // J-type instructions (JAL): rd only\n"
        code += f"  wire is_j_type = ({instr_data}[6:2] == 5'b11011);    // JAL\n"
        code += "  always_comb begin\n"
        code += f"    assume (({instr_data}[1:0] != 2'b11) || !is_j_type ||\n"
        code += f"            ({instr_data}[11:7] <= 5'd{max_reg}));\n"
        code += "  end\n\n"

    # Generate positive constraints from required extensions
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
                        valid_patterns.append((encoding.base_pattern, encoding.base_mask, f"{name} ({ext})"))

        if valid_patterns:
            code += "  // Instruction must match one of these valid patterns (OR of all valid instructions)\n"
            code += "  always_comb begin\n"
            code += f"    assume (({instr_data}[1:0] != 2'b11) || (\n"

            # Generate OR of all valid instruction patterns
            for i, (pattern, mask, desc) in enumerate(valid_patterns):
                is_last = (i == len(valid_patterns) - 1)
                connector = "" if is_last else " ||"
                code += f"      (({instr_data} & 32'h{mask:08x}) == 32'h{pattern:08x}){connector}  // {desc}\n"

            code += "    ));\n"
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
            code += f"    assume (({instr_data}[1:0] != 2'b11) || (({instr_data}[{data_width-1}:0] & {data_width}'h{mask:08x}) != {data_width}'h{pattern:08x}));\n"
            code += f"  end\n\n"

    # Generate constraints for 16-bit compressed instructions
    if patterns_16bit:
        code += "  // 16-bit compressed instruction outlawed patterns\n"
        code += "  // When out of reset and instruction is compressed, these patterns don't occur\n"
        for pattern, mask, desc in patterns_16bit:
            code += f"  // {desc}: Pattern=0x{pattern:04x}, Mask=0x{mask:04x}\n"
            code += f"  always_comb begin\n"
            code += f"    assume (({instr_data}[1:0] == 2'b11) || (({instr_data}[15:0] & 16'h{mask:04x}) != 16'h{pattern:04x}));\n"
            code += f"  end\n\n"

    return code

def main():
    parser = argparse.ArgumentParser(
        description='Generate inline SystemVerilog assumptions from instruction DSL'
    )
    parser.add_argument('input_file', help='DSL file containing instruction rules')
    parser.add_argument('output_file', help='Output SystemVerilog file (inline assumptions)')
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

    # Check if C extension is required
    has_c_ext = has_c_extension_required(ast.rules)
    if has_c_ext:
        print("C extension detected - will auto-expand compressed instructions")

    # Separate rules into required extensions, register constraints, PC constraints, and outlawed patterns
    required_extensions = set()
    register_constraint = None
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
        elif isinstance(rule, PcConstraintRule):
            if pc_constraint is not None:
                print(f"Warning: Multiple PC constraints found, using the last one ({rule.pc_bits} bits)")
            pc_constraint = rule
        elif isinstance(rule, InstructionRule):
            instruction_rules.append(rule)  # Save for dtype processing
            rule_patterns = instruction_rule_to_pattern(rule, has_c_ext)
            patterns.extend(rule_patterns)
        elif isinstance(rule, PatternRule):
            desc = rule.description if rule.description else f"Pattern at line {rule.line}"
            patterns.append((rule.pattern, rule.mask, desc, False))  # PatternRules are always 32-bit

    if required_extensions:
        print(f"Required extensions: {', '.join(sorted(required_extensions))}")
    if register_constraint:
        print(f"Register constraint: x{register_constraint.min_reg}-x{register_constraint.max_reg} ({register_constraint.max_reg - register_constraint.min_reg + 1} registers)")
    if pc_constraint:
        addr_space_kb = (2 ** pc_constraint.pc_bits) // 1024
        print(f"PC constraint: {pc_constraint.pc_bits} bits ({addr_space_kb}KB address space)")
    print(f"Generated {len(patterns)} outlawed patterns")

    # Generate inline assumptions code
    pc_bits = pc_constraint.pc_bits if pc_constraint else None

    print("Generating inline SystemVerilog assumptions...")
    code = generate_inline_assumptions(patterns, required_extensions, register_constraint, pc_bits, config)

    with open(args.output_file, 'w') as f:
        f.write(code)
    print(f"Successfully wrote inline assumptions to {args.output_file}")

if __name__ == '__main__':
    main()
