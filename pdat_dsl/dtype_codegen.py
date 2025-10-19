#!/usr/bin/env python3
"""
Generate SystemVerilog assertions for data type constraints.

This module generates assertions that check operand bit patterns match
the specified data type constraints (width and signedness).
"""

from typing import List, Tuple, Set, Optional
from .parser import InstructionRule, DataType, DataTypeSet

def generate_dtype_check_expr(dtype: DataType, operand_signal: str, width: int = 32) -> str:
    """
    Generate a SystemVerilog expression that checks if an operand matches a data type.

    Args:
        dtype: The data type to check (e.g., i16, u8)
        operand_signal: The signal name (e.g., "multdiv_operand_a_ex")
        width: The full width of the operand (default 32 for RV32)

    Returns:
        A SystemVerilog boolean expression

    Examples:
        i8:  bits [31:8] must be sign-extended from bit 7
             (operand[31:8] == {24{operand[7]}})
        u8:  bits [31:8] must be zero
             (operand[31:8] == 24'b0)
        i16: bits [31:16] must be sign-extended from bit 15
             (operand[31:16] == {16{operand[15]}})
        u16: bits [31:16] must be zero
             (operand[31:16] == 16'b0)
    """
    assert dtype.width < width, f"Data type width {dtype.width} must be less than operand width {width}"

    # Calculate the sign bit position and number of upper bits
    sign_bit = dtype.width - 1
    upper_bits = width - dtype.width

    if dtype.signed:
        # Signed: upper bits must be sign-extended
        # Example: i8 -> operand[31:8] == {24{operand[7]}}
        return f"({operand_signal}[{width-1}:{dtype.width}] == {{{upper_bits}{{{operand_signal}[{sign_bit}]}}}})"
    else:
        # Unsigned: upper bits must be zero
        # Example: u8 -> operand[31:8] == 24'b0
        return f"({operand_signal}[{width-1}:{dtype.width}] == {upper_bits}'b0)"


def generate_dtype_set_check_expr(dtype_set: DataTypeSet, operand_signal: str, width: int = 32) -> str:
    """
    Generate a SystemVerilog expression that checks if an operand matches any type in a set.

    Args:
        dtype_set: The set of data types (possibly negated)
        operand_signal: The signal name
        width: The full width of the operand

    Returns:
        A SystemVerilog boolean expression

    Examples:
        i8 | u8:     (check_i8 || check_u8)
        ~i16:        check_i16  (single type, negated)
        ~(i16 | u16): (check_i16 || check_u16)  (will be negated by caller)
    """
    if len(dtype_set.types) == 0:
        raise ValueError("DataTypeSet cannot be empty")

    # Generate check expressions for each type
    checks = []
    for dtype in sorted(dtype_set.types, key=lambda t: (t.width, not t.signed)):
        check = generate_dtype_check_expr(dtype, operand_signal, width)
        checks.append(check)

    # Combine with OR
    if len(checks) == 1:
        combined = checks[0]
    else:
        combined = "(" + " || ".join(checks) + ")"

    return combined


def generate_instruction_dtype_assertions(rule: InstructionRule,
                                          instr_valid_signal: str = "instr_valid_i",
                                          instr_data_signal: str = "instr_rdata_i") -> List[str]:
    """
    Generate SystemVerilog assertions for data type constraints on an instruction.

    Args:
        rule: The instruction rule with dtype constraints
        instr_valid_signal: Signal indicating instruction is valid
        instr_data_signal: The instruction bits signal

    Returns:
        List of SystemVerilog assertion strings

    The generated assertions check:
    1. The instruction matches the opcode
    2. The operands satisfy the data type constraints
    """
    from .encodings import get_instruction_encoding

    assertions = []

    # Get instruction encoding
    encoding = get_instruction_encoding(rule.name)
    if not encoding:
        return []  # Unknown instruction, skip

    # Check if rule has any dtype constraints
    dtype_constraints = {}
    for constraint in rule.constraints:
        field_name = constraint.field_name
        if field_name == 'dtype' or field_name.endswith('_dtype'):
            dtype_constraints[field_name] = constraint.field_value

    if not dtype_constraints:
        return []  # No dtype constraints

    # For each dtype constraint, generate assertions
    for field_name, dtype_set in dtype_constraints.items():
        if not isinstance(dtype_set, DataTypeSet):
            continue  # Not a dtype constraint

        # Determine which operand(s) this constraint applies to
        if field_name == 'dtype':
            # Applies to all operands
            operand_fields = []
            if 'rd' in encoding.fields:
                operand_fields.append('rd')
            if 'rs1' in encoding.fields:
                operand_fields.append('rs1')
            if 'rs2' in encoding.fields:
                operand_fields.append('rs2')
        elif field_name == 'rd_dtype':
            operand_fields = ['rd']
        elif field_name == 'rs1_dtype':
            operand_fields = ['rs1']
        elif field_name == 'rs2_dtype':
            operand_fields = ['rs2']
        else:
            continue  # Unknown dtype field

        # Map operand fields to actual signals
        # This depends on the microarchitecture
        for operand_field in operand_fields:
            signal_name = get_operand_signal_name(rule.name, operand_field)
            if not signal_name:
                continue

            # Generate the check expression
            check_expr = generate_dtype_set_check_expr(dtype_set, signal_name)

            # Apply negation if needed
            if dtype_set.negated:
                # Negated: we want to ALLOW only these types
                # So we assert that the operand MUST match one of these types
                assertion_condition = check_expr
                semantic = "allow only"
            else:
                # Not negated: we want to FORBID these types
                # So we assert that the operand does NOT match any of these types
                assertion_condition = f"!{check_expr}"
                semantic = "forbid"

            # Create the assertion
            assertion = f"  // {rule.name} {operand_field}: {semantic} {dtype_set}\n"
            assertion += f"  assert property (@(posedge clk_i) disable iff (!rst_ni)\n"
            assertion += f"    ({instr_valid_signal} && is_{rule.name.lower()}_{operand_field}) |-> {assertion_condition}\n"
            assertion += f"  ) else $error(\"{rule.name} {operand_field} violates dtype constraint: {dtype_set}\");\n"

            assertions.append(assertion)

    return assertions


def get_operand_signal_name(instr_name: str, operand_field: str) -> Optional[str]:
    """
    Map instruction name and operand field to the actual signal name in Ibex.

    For Ibex microarchitecture:
    - ALU instructions: alu_operand_a_ex, alu_operand_b_ex
    - MUL/DIV instructions: multdiv_operand_a_ex, multdiv_operand_b_ex
    - LSU stores: lsu_wdata (from rs2)
    - rd (destination): checked at writeback via rf_wdata
    """
    # Classify instruction type
    mul_div_instrs = {'MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'}
    store_instrs = {'SB', 'SH', 'SW', 'SD'}

    if instr_name.upper() in mul_div_instrs:
        # Multiply/Divide unit
        if operand_field == 'rs1':
            return 'multdiv_operand_a_ex'
        elif operand_field == 'rs2':
            return 'multdiv_operand_b_ex'
        elif operand_field == 'rd':
            # Result written back - check at writeback stage
            return 'rf_wdata_wb'  # May need to gate with mult_en
    elif instr_name.upper() in store_instrs:
        # Store instruction
        if operand_field == 'rs2':
            return 'lsu_wdata'
        elif operand_field == 'rs1':
            return 'alu_operand_a_ex'  # Address calculation
    else:
        # Regular ALU instruction
        if operand_field == 'rs1':
            return 'alu_operand_a_ex'
        elif operand_field == 'rs2':
            return 'alu_operand_b_ex'
        elif operand_field == 'rd':
            return 'rf_wdata_wb'

    return None
