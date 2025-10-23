# Example outlawed instruction specification

# Specify which instruction extensions are valid (positive constraints)
# This tells the optimizer that ONLY instructions from these extensions are valid
require RV32I
require RV32M

# Outlaw the ADD instruction (from RV32I extension)
# Even though RV32M is required above, the ADD instruction is outlawed as an example
instruction ADD
