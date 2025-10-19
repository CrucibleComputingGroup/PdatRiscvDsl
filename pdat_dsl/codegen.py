#!/usr/bin/env python3
"""
Generate a SystemVerilog assertion module from instruction DSL.

This script:
1. Parses the DSL file containing instruction rules
2. Converts high-level instruction rules to pattern/mask pairs
3. Generates a SystemVerilog module with assertions
"""

import sys
import argparse
from typing import List, Tuple, Set, Optional
from .parser import parse_dsl, InstructionRule, PatternRule, FieldConstraint, RequireRule, RegisterConstraintRule
from .encodings import (
    get_instruction_encoding, parse_register, set_field, create_field_mask,
    get_extension_instructions
)

def instruction_rule_to_pattern(rule: InstructionRule) -> List[Tuple[int, int, str]]:
    """
    Convert an InstructionRule to one or more (pattern, mask, description) tuples.

    Returns a list because some rules with wildcards might expand to multiple patterns.
    """
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

    return [(pattern, mask, desc)]

def generate_sv_module(patterns: List[Tuple[int, int, str]], module_name: str = "instr_outlawed_checker") -> str:
    """Generate SystemVerilog module with pattern-matching assertions."""

    sv_code = f"""// Auto-generated instruction pattern checker module
// This module checks that certain instruction patterns are never present in the pipeline

module {module_name} (
  input logic        clk_i,
  input logic        rst_ni,
  input logic        instr_valid_i,
  input logic [31:0] instr_rdata_i,
  input logic        instr_is_compressed_i
);

"""

    # Group by width (assume all 32-bit for now, can add 16-bit later)
    patterns_32 = [(p, m, d) for p, m, d in patterns if m <= 0xFFFFFFFF]

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

def generate_inline_assumptions(patterns, required_extensions: Set[str] = None,
                               register_constraint: Optional[RegisterConstraintRule] = None):
    """Generate assumptions to inject directly into ID stage (no separate module)."""

    code = "\n  // ========================================\n"
    code += "  // Auto-generated instruction constraints\n"
    code += "  // ========================================\n\n"

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
        code += "  wire is_r_type = (instr_rdata_i[6:2] == 5'b01100) ||  // OP (ADD, SUB, etc.)\n"
        code += "                   (instr_rdata_i[6:2] == 5'b01110) ||  // OP-32 (ADDW, SUBW, etc.)\n"
        code += "                   (instr_rdata_i[6:2] == 5'b10100) ||  // OP-FP\n"
        code += "                   (instr_rdata_i[6:2] == 5'b10110);    // OP-V\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || instr_is_compressed_i || !is_r_type ||\n"
        code += f"            ((instr_rdata_i[11:7] <= 5'd{max_reg}) &&   // rd\n"
        code += f"             (instr_rdata_i[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             (instr_rdata_i[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # I-type (loads, JALR, OP-IMM): rd, rs1
        code += "  // I-type instructions (LOAD, OP-IMM, JALR): rd, rs1\n"
        code += "  wire is_i_type = (instr_rdata_i[6:2] == 5'b00000) ||  // LOAD\n"
        code += "                   (instr_rdata_i[6:2] == 5'b00100) ||  // OP-IMM\n"
        code += "                   (instr_rdata_i[6:2] == 5'b00110) ||  // OP-IMM-32\n"
        code += "                   (instr_rdata_i[6:2] == 5'b11001);    // JALR\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || instr_is_compressed_i || !is_i_type ||\n"
        code += f"            ((instr_rdata_i[11:7] <= 5'd{max_reg}) &&   // rd\n"
        code += f"             (instr_rdata_i[19:15] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # S-type (stores): rs1, rs2 (no rd - bits [11:7] are immediate)
        code += "  // S-type instructions (STORE): rs1, rs2 (no rd)\n"
        code += "  wire is_s_type = (instr_rdata_i[6:2] == 5'b01000);    // STORE\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || instr_is_compressed_i || !is_s_type ||\n"
        code += f"            ((instr_rdata_i[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             (instr_rdata_i[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # B-type (branches): rs1, rs2 (no rd)
        code += "  // B-type instructions (BRANCH): rs1, rs2 (no rd)\n"
        code += "  wire is_b_type = (instr_rdata_i[6:2] == 5'b11000);    // BRANCH\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || instr_is_compressed_i || !is_b_type ||\n"
        code += f"            ((instr_rdata_i[19:15] <= 5'd{max_reg}) &&  // rs1\n"
        code += f"             (instr_rdata_i[24:20] <= 5'd{max_reg})));\n"
        code += "  end\n\n"

        # U-type (LUI, AUIPC): rd only
        code += "  // U-type instructions (LUI, AUIPC): rd only\n"
        code += "  wire is_u_type = (instr_rdata_i[6:2] == 5'b01101) ||  // LUI\n"
        code += "                   (instr_rdata_i[6:2] == 5'b00101);    // AUIPC\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || instr_is_compressed_i || !is_u_type ||\n"
        code += f"            (instr_rdata_i[11:7] <= 5'd{max_reg}));\n"
        code += "  end\n\n"

        # J-type (JAL): rd only
        code += "  // J-type instructions (JAL): rd only\n"
        code += "  wire is_j_type = (instr_rdata_i[6:2] == 5'b11011);    // JAL\n"
        code += "  always_comb begin\n"
        code += f"    assume (!rst_ni || instr_is_compressed_i || !is_j_type ||\n"
        code += f"            (instr_rdata_i[11:7] <= 5'd{max_reg}));\n"
        code += "  end\n\n"

    # Generate positive constraints from required extensions
    if required_extensions:
        code += "  // Positive constraint: instruction must be from required extensions\n"
        code += f"  // Required: {', '.join(sorted(required_extensions))}\n"

        # Collect all valid instruction patterns from required extensions
        valid_patterns = []
        for ext in required_extensions:
            ext_instrs = get_extension_instructions(ext)
            if ext_instrs:
                for name, encoding in ext_instrs.items():
                    valid_patterns.append((encoding.base_pattern, encoding.base_mask, f"{name} ({ext})"))

        if valid_patterns:
            code += "  // Instruction must match one of these valid patterns (OR of all valid instructions)\n"
            code += "  always_comb begin\n"
            code += "    assume (!rst_ni || instr_is_compressed_i || (\n"

            # Generate OR of all valid instruction patterns
            for i, (pattern, mask, desc) in enumerate(valid_patterns):
                is_last = (i == len(valid_patterns) - 1)
                connector = "" if is_last else " ||"
                code += f"      ((instr_rdata_i & 32'h{mask:08x}) == 32'h{pattern:08x}){connector}  // {desc}\n"

            code += "    ));\n"
            code += "  end\n\n"

    if not patterns:
        if not required_extensions:
            code += "  // No instruction constraints specified\n\n"
        return code

    code += "  // Instruction encoding constraints (unconditional form for ABC)\n"
    code += "  // When out of reset and instruction is uncompressed, these patterns don't occur\n"
    code += "  // Written as: assume (!rst_ni || instr_is_compressed_i || pattern_mismatch)\n"
    for pattern, mask, desc in patterns:
        code += f"  // {desc}: Pattern=0x{pattern:08x}, Mask=0x{mask:08x}\n"
        code += f"  always_comb begin\n"
        code += f"    assume (!rst_ni || instr_is_compressed_i || ((instr_rdata_i & 32'h{mask:08x}) != 32'h{pattern:08x}));\n"
        code += f"  end\n\n"

    return code

def main():
    parser = argparse.ArgumentParser(
        description='Generate SystemVerilog assertion module from instruction DSL'
    )
    parser.add_argument('input_file', help='DSL file containing instruction rules')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--inline', action='store_true',
                       help='Generate inline assumptions code (not a module)')
    parser.add_argument('-m', '--module-name', default='instr_outlawed_checker',
                       help='Name of generated module (default: instr_outlawed_checker)')
    parser.add_argument('-b', '--bind-file',
                       help='Output bind file (default: <output_file_base>_bind.sv)')

    args = parser.parse_args()

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

    # Separate rules into required extensions, register constraints, and outlawed patterns
    required_extensions = set()
    register_constraint = None
    patterns = []

    for rule in ast.rules:
        if isinstance(rule, RequireRule):
            required_extensions.add(rule.extension)
        elif isinstance(rule, RegisterConstraintRule):
            if register_constraint is not None:
                print(f"Warning: Multiple register constraints found, using the last one (x{rule.min_reg}-x{rule.max_reg})")
            register_constraint = rule
        elif isinstance(rule, InstructionRule):
            rule_patterns = instruction_rule_to_pattern(rule)
            patterns.extend(rule_patterns)
        elif isinstance(rule, PatternRule):
            desc = rule.description if rule.description else f"Pattern at line {rule.line}"
            patterns.append((rule.pattern, rule.mask, desc))

    if required_extensions:
        print(f"Required extensions: {', '.join(sorted(required_extensions))}")
    if register_constraint:
        print(f"Register constraint: x{register_constraint.min_reg}-x{register_constraint.max_reg} ({register_constraint.max_reg - register_constraint.min_reg + 1} registers)")
    print(f"Generated {len(patterns)} outlawed patterns")

    # Generate code
    if args.inline:
        # Generate inline assumptions code only
        code = generate_inline_assumptions(patterns, required_extensions, register_constraint)
        with open(args.output_file, 'w') as f:
            f.write(code)
        print(f"Successfully wrote inline assumptions to {args.output_file}")
    else:
        # Generate full module + bind file
        # Determine bind file name
        if args.bind_file:
            bind_file = args.bind_file
        else:
            # Default: replace .sv with _bind.sv
            if args.output_file.endswith('.sv'):
                bind_file = args.output_file[:-3] + '_bind.sv'
            else:
                bind_file = args.output_file + '_bind.sv'

        print(f"Generating SystemVerilog module '{args.module_name}'...")
        sv_code = generate_sv_module(patterns, args.module_name)

        # Write checker module
        with open(args.output_file, 'w') as f:
            f.write(sv_code)
        print(f"Successfully wrote {args.output_file}")

        # Generate and write bind file
        bind_code = generate_bind_file(args.module_name)
        with open(bind_file, 'w') as f:
            f.write(bind_code)
        print(f"Successfully wrote {bind_file}")

if __name__ == '__main__':
    main()
