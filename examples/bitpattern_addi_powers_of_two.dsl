# Example: Restrict ADDI immediate to powers of 2
# Demonstrates bit patterns for specific immediate values

require RV32I

# ADDI has a 12-bit signed immediate in bits [31:20]

# Allow only ADDI with immediate = 1 (0x001)
instruction ADDI { imm = 12'b000000000001 }

# Allow only ADDI with immediate = 2 (0x002)
instruction ADDI { imm = 12'b000000000010 }

# Allow only ADDI with immediate = 4 (0x004)
instruction ADDI { imm = 12'b000000000100 }

# Allow only ADDI with immediate = 8 (0x008)
instruction ADDI { imm = 12'b000000001000 }

# Result (in outlawing semantics): These specific values are forbidden
# All other immediate values are allowed

# For positive constraint (allow ONLY these), use:
# allow instruction ADDI { imm = 12'b000000000001 }
# allow instruction ADDI { imm = 12'b000000000010 }
# allow instruction ADDI { imm = 12'b000000000100 }
# allow instruction ADDI { imm = 12'b000000001000 }
# (Only powers of 2 from 1-8 would be allowed)
