# Example: Allow only specific shift amounts
# Demonstrates multiple bit pattern constraints on the same instruction

require RV32I

# Allow ONLY shift by 1 (no other shift amounts)
# These rules collectively forbid all shift amounts except 1
instruction SLLI { imm = 5'b00000 }  # Forbid shift by 0
instruction SLLI { imm = 5'b00010 }  # Forbid shift by 2
instruction SLLI { imm = 5'b00011 }  # Forbid shift by 3
instruction SLLI { imm = 5'b001xx }  # Forbid shift by 4-7
instruction SLLI { imm = 5'b01xxx }  # Forbid shift by 8-15
instruction SLLI { imm = 5'b1xxxx }  # Forbid shift by 16-31

# Result: Only SLLI with imm = 5'b00001 (shift by 1) is allowed

# Note: With the 'allow' keyword, this becomes much simpler:
# allow instruction SLLI { imm = 5'b00001 }  # Allow ONLY shift by 1
