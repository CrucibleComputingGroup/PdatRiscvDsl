# RV32IM with 64KB instruction memory constraint
# This example demonstrates using require_pc_bits to restrict the PC
# to a limited address space (useful for formal verification and synthesis)

# Require base instruction sets
require RV32I
require RV32M

# Constrain PC to 16 bits (64KB address space)
# This forces PC[31:16] to be zero, limiting instruction memory to first 64KB
require_pc_bits 16

# Optionally outlaw specific instructions
# instruction DIV
# instruction REM
