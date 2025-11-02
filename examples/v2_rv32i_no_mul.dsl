version 2

# Example: RV32I without multiplication (v2 syntax)
# Equivalent to v1: require RV32I; require RV32M; instruction MUL; ...

include RV32I
include RV32M

# Remove all multiplication instructions
forbid MUL
forbid MULH
forbid MULHSU
forbid MULHU
forbid DIV
forbid DIVU
forbid REM
forbid REMU

# Result: RV32I + RV32M minus all M-extension instructions = RV32I only
