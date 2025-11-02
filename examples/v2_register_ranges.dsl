version 2

# Example: Limit specific instruction to use only low registers
# Demonstrates range constraints with BDD optimization

# Include only ADD with rd in x0-x15
include ADD {rd in x0-x15}

# Include only SUB with both operands in x0-x15
include SUB {rd in x0-x15, rs1 in x0-x15, rs2 in x0-x15}

# Result: Very compact SystemVerilog with just 2 patterns
# BDD automatically optimizes to check high bits instead of enumerating all combinations
