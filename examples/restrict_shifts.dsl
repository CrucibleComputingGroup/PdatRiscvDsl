# Example: Restrict shift instructions to small shift amounts
# This demonstrates using bit patterns to constrain immediate field values

# Specify valid instruction set
require RV32I

# Restrict SLLI (logical left shift immediate) to shifts by 0-3 only
# RV32 shift instructions use I-type format with a 12-bit immediate field
# For shifts, bits [31:25] are funct7 (fixed), bits [24:20] are the 5-bit shift amount
# The 'imm' field in the encoding is 12 bits, so we need to specify all 12 bits
# Pattern: 0000000 (7 bits funct7 for SLLI) + 000xx (5 bits shamt 0-3)

# SLLI: funct7=0x00 (0000000), shamt=0-3
instruction SLLI { imm = 12'b000000000000 }  # Forbid shift by 0
instruction SLLI { imm = 12'b000000000001 }  # Forbid shift by 1
instruction SLLI { imm = 12'b000000000010 }  # Forbid shift by 2
instruction SLLI { imm = 12'b000000000011 }  # Forbid shift by 3

# SRLI: funct7=0x00 (0000000), shamt=0-3
instruction SRLI { imm = 12'b000000000000 }  # Forbid shift by 0
instruction SRLI { imm = 12'b000000000001 }  # Forbid shift by 1
instruction SRLI { imm = 12'b000000000010 }  # Forbid shift by 2
instruction SRLI { imm = 12'b000000000011 }  # Forbid shift by 3

# SRAI: funct7=0x20 (0100000), shamt=0-3
instruction SRAI { imm = 12'b010000000000 }  # Forbid shift by 0
instruction SRAI { imm = 12'b010000000001 }  # Forbid shift by 1
instruction SRAI { imm = 12'b010000000010 }  # Forbid shift by 2
instruction SRAI { imm = 12'b010000000011 }  # Forbid shift by 3

# Note: In an outlawing framework, these rules forbid shifts by amounts 0-3
# All other shift amounts (4-31) remain allowed
#
# To ALLOW only shifts 0-3 and forbid all others, use the 'allow' keyword:
# allow instruction SLLI { imm = 12'b00000000000x }  # Allow shifts 0-1
# (The 'allow' keyword inverts the semantics - not yet fully implemented)
