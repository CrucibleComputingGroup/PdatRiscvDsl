#!/usr/bin/env python3
"""
Generate SystemVerilog assertions for data type constraints.

This module generates assertions that check operand bit patterns match
the specified data type constraints (width and signedness).
"""

from .parser import DataType, DataTypeSet

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


# Note: Data type constraint code generation for full instruction assertions
# has been removed. The actual implementation is in codegen.py which properly
# uses CoreConfig to get signal names. The functions below (generate_dtype_check_expr
# and generate_dtype_set_check_expr) are still used by codegen.py to generate
# the boolean expressions for data type checking.
