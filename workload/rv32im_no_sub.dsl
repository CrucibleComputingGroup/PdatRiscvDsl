# Example outlawed instruction specification

# Specify which instruction extensions are valid (positive constraints)
# This tells the optimizer that ONLY instructions from these extensions are valid
require RV32I
require RV32M

# Outlaw the SUB instruction (from RV32I extension)
# Even though RV32I is required above, the SUB instruction is specifically outlawed
instruction SUB
