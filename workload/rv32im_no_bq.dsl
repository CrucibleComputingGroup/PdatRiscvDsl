# Example outlawed instruction specification

# Specify which instruction extensions are valid (positive constraints)
# This tells the optimizer that ONLY instructions from these extensions are valid
require RV32I
require RV32M

# Outlaw all branch instructions (BNE, BLT, BGE, BLTU, BGEU)
# Even though RV32I is required above, these specific branch instructions are outlawed
instruction BNE
instruction BLT
instruction BGE
instruction BLTU
instruction BGEU
