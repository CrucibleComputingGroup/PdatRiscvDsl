# RV32IC Example - RV32I with Compressed Instructions
# This demonstrates auto-expansion: when C extension is required,
# outlawing an instruction like "MUL" will also outlaw "C.MUL" if it exists.

# Require base instruction set and compressed extension
require RV32I
require RV32C

# Outlaw all M extension instructions
# Note: Since RV32C is required, this will automatically also outlaw
# compressed versions if they exist (though M extension doesn't have
# compressed equivalents, this is just an example)
instruction MUL
instruction MULH
instruction MULHSU
instruction MULHU
instruction DIV
instruction DIVU
instruction REM
instruction REMU

# You can also explicitly outlaw specific compressed instructions
# instruction C.ADD
# instruction C.MV

# Example: constrain register usage
# require_registers x0-x15
