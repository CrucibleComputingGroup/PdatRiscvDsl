# Example: Data Type Constraints
# This demonstrates specifying data width and signedness constraints
# for RISC-V instructions

# Specify valid instruction set
require RV32I
require RV32M

# ============================================================================
# Instruction-Level Data Type Constraints
# ============================================================================
# These constraints apply to all operands of an instruction.
# Useful when all operands are treated uniformly (e.g., all 16-bit signed).

# Without negation (~): FORBID instructions using these types
# MUL: forbid when using 16-bit signed
instruction MUL { dtype = i16 }

# MULH: forbid when using 8-bit signed
instruction MULH { dtype = i8 }

# Multiple types: forbid when using 8-bit OR 16-bit signed
instruction DIV { dtype = i8 | i16 }

# With negation (~): ALLOW ONLY these types (forbid all others)
# DIVU: allow only 8-bit and 16-bit unsigned (forbid i32, i64, u32, u64)
# Note: must include u8 when allowing u16 (can't distinguish u8 from u16 when value < 256)
instruction DIVU { dtype = ~(u8 | u16) }

# REM: allow only 8-bit types signed or unsigned (forbid everything wider)
instruction REM { dtype = ~(i8 | u8) }

# REMU: allow only 8-bit and 16-bit types (both signed and unsigned)
# Note: when allowing i16/u16, must also allow i8/u8
instruction REMU { dtype = ~(i8 | u8 | i16 | u16) }

# ============================================================================
# Per-Operand Data Type Constraints
# ============================================================================
# These constraints specify types for individual operands (rd, rs1, rs2).
# Useful for mixed-width operations or different signedness requirements.

# MULHU: unsigned multiplication, upper bits
# All operands are unsigned 16-bit
instruction MULHU { rd_dtype = u16, rs1_dtype = u16, rs2_dtype = u16 }

# Mixed signedness example
# Destination and rs1 are signed 8-bit, rs2 can be signed or unsigned 8-bit
instruction MULHSU { rd_dtype = i8, rs1_dtype = i8, rs2_dtype = i8 | u8 }

# Different widths example
# This shows a case where destination might be wider, but sources are constrained
# (Note: semantically this may not make sense for all instructions, just demonstrating syntax)
instruction ADD { rs1_dtype = i8, rs2_dtype = i8 }

# ============================================================================
# Combining with Register Constraints
# ============================================================================
# You can combine data type constraints with register field constraints

# MUL with specific register and data type constraints
instruction MUL { rd = x5, rs1_dtype = i16, rs2_dtype = i16 }

# DIV with wildcard register and data type
instruction DIVU { rd = *, rs1_dtype = u8, rs2_dtype = u8 }

# ============================================================================
# Data Type Semantics
# ============================================================================
# i8:  8-bit signed   (-128 to 127) - sign-extended to full register width
# u8:  8-bit unsigned (0 to 255)    - zero-extended to full register width
# i16: 16-bit signed  (-32768 to 32767) - sign-extended
# u16: 16-bit unsigned (0 to 65535)     - zero-extended
# i32: 32-bit signed   (full register on RV32)
# u32: 32-bit unsigned (full register on RV32)
# i64: 64-bit signed   (for RV64)
# u64: 64-bit unsigned (for RV64)

# ============================================================================
# Type Expression Syntax
# ============================================================================
# The pipe operator (|) allows specifying multiple types in a union:
#   dtype = i8 | u8 | i16
#
# The negation prefix (~) inverts the meaning (can only appear at start):
#   dtype = i8       → forbid when using i8
#   dtype = ~i8      → forbid when NOT using i8 (i.e., allow only i8)
#   dtype = i8 | u8  → forbid when using i8 OR u8
#   dtype = ~(i8 | u8) → forbid when NOT using i8 or u8 (i.e., allow only i8 or u8)
#
# Note: The ~ is a type expression prefix, not a general operator.
# It can only appear once at the very beginning of the expression.
#
# Valid:   ~i16, ~(i16 | u16), ~(i8 | u8 | i16 | u16)
# Invalid: ~~i16, i16 | ~u16, ~i16 | ~u16

# ============================================================================
# Use Cases
# ============================================================================
# 1. Hardware Optimization: Indicate that multiply unit only needs 16-bit inputs,
#    allowing synthesis tools to optimize away upper bits
#
# 2. Formal Verification: Generate assertions that operands are within
#    specified bit ranges
#
# 3. Power Analysis: Understand data switching activity for power estimation
#
# 4. Code Generation: Constrain compiler to only use specific data types,
#    ensuring optimized code paths
