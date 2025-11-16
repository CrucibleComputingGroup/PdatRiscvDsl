# Example outlawed instruction specification

# Specify which instruction extensions are valid (positive constraints)
# This tells the optimizer that ONLY instructions from these extensions are valid
require RV32I

# Outlaw ADD and ADDI instructions (addition instructions from RV32I extension)
# Even though RV32I is required above, these specific addition instructions are outlawed
instruction ADD
instruction ADDI
