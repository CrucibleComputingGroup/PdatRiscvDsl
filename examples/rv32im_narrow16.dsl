# RV32IM with 16-bit maximum data width constraint
# This restricts ALL arithmetic instructions to operate only on 8-bit and 16-bit data types
# Useful for reducing hardware area by implementing only narrow multipliers/ALUs

# Require base instruction sets
require RV32I
require RV32M

# ============================================================================
# Data Type Constraint Strategy
# ============================================================================
# Using negated type sets to ALLOW ONLY narrow types (8-bit and 16-bit)
# This forbids 32-bit operations, enabling:
# - Smaller multiplier/divider units (16-bit instead of 32-bit)
# - Reduced ALU area
# - Synthesis optimization based on narrow operand assumptions
#
# Syntax: dtype = ~(i8 | u8 | i16 | u16)
# Meaning: Allow ONLY these types, forbid everything else (i32, u32, i64, u64)

# ============================================================================
# RV32M Instructions (Multiply/Divide)
# ============================================================================
# These are the primary targets for area reduction

instruction MUL    { dtype = ~(i8 | u8 | i16 | u16) }
instruction MULH   { dtype = ~(i8 | u8 | i16 | u16) }
instruction MULHSU { dtype = ~(i8 | u8 | i16 | u16) }
instruction MULHU  { dtype = ~(i8 | u8 | i16 | u16) }
instruction DIV    { dtype = ~(i8 | u8 | i16 | u16) }
instruction DIVU   { dtype = ~(i8 | u8 | i16 | u16) }
instruction REM    { dtype = ~(i8 | u8 | i16 | u16) }
instruction REMU   { dtype = ~(i8 | u8 | i16 | u16) }

# ============================================================================
# RV32I ALU Instructions (Register-Register)
# ============================================================================

instruction ADD  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SUB  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SLL  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SRL  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SRA  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SLT  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SLTU { dtype = ~(i8 | u8 | i16 | u16) }
instruction XOR  { dtype = ~(i8 | u8 | i16 | u16) }
instruction OR   { dtype = ~(i8 | u8 | i16 | u16) }
instruction AND  { dtype = ~(i8 | u8 | i16 | u16) }

# ============================================================================
# RV32I ALU-Immediate Instructions
# ============================================================================

instruction ADDI  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SLTI  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SLTIU { dtype = ~(i8 | u8 | i16 | u16) }
instruction XORI  { dtype = ~(i8 | u8 | i16 | u16) }
instruction ORI   { dtype = ~(i8 | u8 | i16 | u16) }
instruction ANDI  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SLLI  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SRLI  { dtype = ~(i8 | u8 | i16 | u16) }
instruction SRAI  { dtype = ~(i8 | u8 | i16 | u16) }

# ============================================================================
# Expected Hardware Benefits
# ============================================================================
# - Multiplier: 16x16 instead of 32x32 (approximately 4x area reduction)
# - Divider: 16-bit instead of 32-bit (significant area/timing improvement)
# - ALU: Can optimize assuming upper 16 bits are sign/zero extended
# - Register file: Upper bits may be optimized away in some paths
#
# For formal verification:
# - Reduced state space (operands bounded to 16 bits)
# - Faster convergence of proofs
# - Can prove properties about narrow arithmetic
