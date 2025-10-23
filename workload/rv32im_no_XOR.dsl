# Example outlawed instruction specification

# Specify which instruction extensions are valid (positive constraints)
# This tells the optimizer that ONLY instructions from these extensions are valid
require RV32I
require RV32M

# Outlaw the XOR instruction (RV32I extension)
# Even though RV32I is required above, this specific instruction is outlawed
instruction XOR
