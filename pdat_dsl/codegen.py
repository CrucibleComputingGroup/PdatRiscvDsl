#!/usr/bin/env python3
"""
Generate a SystemVerilog assertion module from instruction DSL.

This script:
1. Parses the DSL file containing instruction rules
2. Converts high-level instruction rules to pattern/mask pairs
3. Generates a SystemVerilog module with assertions
4. Generates data type constraint assertions for operand bit patterns
"""

import sys
import argparse
from typing import List, Tuple, Set, Optional
from .parser import parse_dsl, InstructionRule, PatternRule, FieldConstraint, RequireRule, RegisterConstraintRule, DataTypeSet, TimingConstraintRule
from .encodings import (
    get_instruction_encoding, parse_register, set_field, create_field_mask,
    get_extension_instructions
)
from .dtype_codegen import generate_dtype_check_expr, generate_dtype_set_check_expr

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
            p, m, d, is_compressed = item
            if m <= 0xFFFFFFFF:
                patterns_32.append((p, m, d))
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

def generate_dtype_assertions(rules: List[InstructionRule]) -> str:
    """
    Generate SystemVerilog assertions for data type constraints.

    Returns SystemVerilog code to be added to the checker module.
    """
    code = ""

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

            # Determine which operands to check
            if field_name == 'dtype':
                # Apply to all operands
                operand_fields = []
                if 'rs1' in encoding.fields:
                    operand_fields.append(('rs1', 'multdiv_operand_a_ex_i' if rule.name.upper() in {'MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'} else 'alu_operand_a_ex_i'))
                if 'rs2' in encoding.fields:
                    operand_fields.append(('rs2', 'multdiv_operand_b_ex_i' if rule.name.upper() in {'MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'} else 'alu_operand_b_ex_i'))
            elif field_name == 'rs1_dtype':
                operand_fields = [('rs1', 'multdiv_operand_a_ex_i' if rule.name.upper() in {'MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'} else 'alu_operand_a_ex_i')]
            elif field_name == 'rs2_dtype':
                operand_fields = [('rs2', 'multdiv_operand_b_ex_i' if rule.name.upper() in {'MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'} else 'alu_operand_b_ex_i')]
            elif field_name == 'rd_dtype':
                # Skip rd for now - would need writeback stage signals
                continue
            else:
                continue

            # Generate check for each operand
            for operand_name, signal_name in operand_fields:
                check_expr = generate_dtype_set_check_expr(dtype_set, signal_name)

                # Create instruction match condition
                instr_match = f"((instr_rdata_i & 32'h{encoding.base_mask:08x}) == 32'h{encoding.base_pattern:08x})"

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
                code += f"  always_comb begin\n"
                code += f"    if (rst_ni && instr_valid_i && {instr_match}) begin\n"
                code += f"      assume ({condition});\n"
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


def generate_inline_assumptions(patterns, required_extensions: Set[str] = None,
                               register_constraint: Optional[RegisterConstraintRule] = None):
    """Generate assumptions to inject directly into ID stage (no separate module)."""

    code = "\n  // ========================================\n"
    code += "  // Auto-generated instruction constraints\n"
    code += "  // ========================================\n\n"

    # Add compression bit consistency check
    code += "  // Compression bit consistency: instr_is_compressed_i must match encoding bits[1:0]\n"
    code += "  // Compressed instructions have bits[1:0] != 11, uncompressed have bits[1:0] == 11\n"
    code += "  always_comb begin\n"
    code += "    assume (!rst_ni || (instr_is_compressed_i == (instr_rdata_i[1:0] != 2'b11)));\n"
    code += "  end\n\n"

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
            code += f"    assume (!rst_ni || instr_is_compressed_i || ((instr_rdata_i[31:0] & 32'h{mask:08x}) != 32'h{pattern:08x}));\n"
            code += f"  end\n\n"

    # Generate constraints for 16-bit compressed instructions
    if patterns_16bit:
        code += "  // 16-bit compressed instruction outlawed patterns\n"
        code += "  // When out of reset and instruction is compressed, these patterns don't occur\n"
        for pattern, mask, desc in patterns_16bit:
            code += f"  // {desc}: Pattern=0x{pattern:04x}, Mask=0x{mask:04x}\n"
            code += f"  always_comb begin\n"
            code += f"    assume (!rst_ni || !instr_is_compressed_i || ((instr_rdata_i[15:0] & 16'h{mask:04x}) != 16'h{pattern:04x}));\n"
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

    # Check if C extension is required
    has_c_ext = has_c_extension_required(ast.rules)
    if has_c_ext:
        print("C extension detected - will auto-expand compressed instructions")

    # Separate rules into required extensions, register constraints, timing constraints, and outlawed patterns
    required_extensions = set()
    register_constraint = None
    timing_constraint = None
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
    if timing_constraint:
        timing_parts = []
        for param, value in timing_constraint.items():
            timing_parts.append(f"{param}={value}")
        print(f"Timing constraints: {', '.join(timing_parts)}")
    print(f"Generated {len(patterns)} outlawed patterns")

    # Generate code
    if args.inline:
        # Generate inline assumptions code only
        code = generate_inline_assumptions(patterns, required_extensions, register_constraint)
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

        # Generate data type assertions
        dtype_assertions = generate_dtype_assertions(instruction_rules)

        sv_code = generate_sv_module(patterns, args.module_name, dtype_assertions)

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
