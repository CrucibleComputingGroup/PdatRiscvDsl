# RV32IM with 8-bit maximum data width constraint
# This restricts ALL arithmetic instructions to operate only on 8-bit data types
# Useful for minimal hardware implementations (8-bit microcontrollers, embedded systems)

# Require base instruction sets
require RV32I
require RV32M

# ============================================================================
# Data Type Constraint Strategy
# ============================================================================
# Using negated type sets to ALLOW ONLY 8-bit types
# This forbids 16-bit and 32-bit operations, enabling:
# - Minimal multiplier/divider units (8-bit only)
# - Smallest possible ALU area
# - Maximum synthesis optimization based on 8-bit operand assumptions
#
# Syntax: dtype = ~(i8 | u8)
# Meaning: Allow ONLY these types, forbid everything else (i16, u16, i32, u32, i64, u64)

# ============================================================================
# RV32M Instructions (Multiply/Divide)
# ============================================================================
# 8-bit multiplier: 8x8=16 bit result (vs 32x32=64 bit)
# Area reduction: approximately 16x smaller than 32-bit multiplier

instruction MUL    { dtype = ~(i8 | u8) }
instruction MULH   { dtype = ~(i8 | u8) }
instruction MULHSU { dtype = ~(i8 | u8) }
instruction MULHU  { dtype = ~(i8 | u8) }
instruction DIV    { dtype = ~(i8 | u8) }
instruction DIVU   { dtype = ~(i8 | u8) }
instruction REM    { dtype = ~(i8 | u8) }
instruction REMU   { dtype = ~(i8 | u8) }

# ============================================================================
# RV32I ALU Instructions (Register-Register)
# ============================================================================

instruction ADD  { dtype = ~(i8 | u8) }
instruction SUB  { dtype = ~(i8 | u8) }
instruction SLL  { dtype = ~(i8 | u8) }
instruction SRL  { dtype = ~(i8 | u8) }
instruction SRA  { dtype = ~(i8 | u8) }
instruction SLT  { dtype = ~(i8 | u8) }
instruction SLTU { dtype = ~(i8 | u8) }
instruction XOR  { dtype = ~(i8 | u8) }
instruction OR   { dtype = ~(i8 | u8) }
instruction AND  { dtype = ~(i8 | u8) }

# ============================================================================
# RV32I ALU-Immediate Instructions
# ============================================================================

instruction ADDI  { dtype = ~(i8 | u8) }
instruction SLTI  { dtype = ~(i8 | u8) }
instruction SLTIU { dtype = ~(i8 | u8) }
instruction XORI  { dtype = ~(i8 | u8) }
instruction ORI   { dtype = ~(i8 | u8) }
instruction ANDI  { dtype = ~(i8 | u8) }
instruction SLLI  { dtype = ~(i8 | u8) }
instruction SRLI  { dtype = ~(i8 | u8) }
instruction SRAI  { dtype = ~(i8 | u8) }

# ============================================================================
# Expected Hardware Benefits
# ============================================================================
# Compared to full 32-bit implementation:
# - Multiplier: 8x8 instead of 32x32 (approximately 16x area reduction)
# - Divider: 8-bit instead of 32-bit (major area/timing improvement)
# - ALU: Can optimize assuming upper 24 bits are sign/zero extended
# - Overall: Suitable for ultra-low-area embedded microcontrollers
#
# Compared to 16-bit configuration (rv32im_narrow16.dsl):
# - Multiplier: 8x8 instead of 16x16 (approximately 4x additional reduction)
# - Total area savings: ~60-70% vs narrow16, ~90-95% vs full 32-bit
#
# Use cases:
# - 8-bit sensor nodes
# - Minimal RISC-V implementations
# - Power-constrained embedded systems
# - Formal verification with minimal state space
