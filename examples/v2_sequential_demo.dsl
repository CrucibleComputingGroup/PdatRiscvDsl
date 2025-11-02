version 2

# Example: Demonstrating sequential semantics
# This shows how order matters in v2

# Include full RV32I (40 instructions)
include RV32I

# Remove ADD instruction
forbid ADD

# Add back ADD, but only with rd=x0 (effectively NOP)
include ADD {rd = x0}

# Remove ADDI
forbid ADDI

# Add back ADDI with specific immediates
include ADDI {imm = 12'b000000000001}  # ADDI with imm=1
include ADDI {imm = 12'b000000000010}  # ADDI with imm=2
include ADDI {imm = 12'b000000000100}  # ADDI with imm=4

# Result:
# - RV32I base set
# - ADD only allowed when rd=x0
# - ADDI only allowed with imm ∈ {1, 2, 4}
# - All other instructions unchanged
