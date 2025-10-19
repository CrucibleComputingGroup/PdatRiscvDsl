# Example outlawed instruction specification

# Specify which instruction extensions are valid (positive constraints)
# This tells the optimizer that ONLY instructions from these extensions are valid
require RV32I
require RV32M

# Outlaw all multiply and divide instructions (RV32M extension)
# Even though RV32M is required above, these specific instructions are outlawed
instruction BNE
instruction BLT
instruction BGE
instruction BLTU
instruction BGEU
