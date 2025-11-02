# Example: Allow ONLY specific shift amounts using 'allow' keyword
# This demonstrates positive constraints with bit patterns

require RV32I

# Allow ONLY SLLI with shift amounts 0-3 (forbid all others: 4-31)
# Using 'allow' keyword inverts the semantics
#
# Pattern: 12'b0000000000xx
# - Upper 7 bits (funct7): 0000000 (required for SLLI)
# - Lower 5 bits (shamt): 000xx (matches 0, 1, 2, 3)
#
# With 'allow' semantics: This pattern is the ONLY one allowed
# All other SLLI patterns (shamt 4-31) are implicitly forbidden
allow instruction SLLI { imm = 12'b00000000000x }  # Allow shifts 0-1
allow instruction SLLI { imm = 12'b000000000010 }  # Allow shift 2
allow instruction SLLI { imm = 12'b000000000011 }  # Allow shift 3

# Result: Only SLLI with shamt in {0, 1, 2, 3} is allowed
# SLLI with shamt in {4, 5, ..., 31} is forbidden

# Alternative: Using a pattern with don't-cares to cover multiple values
# allow instruction SLLI { imm = 12'b00000000000x }  # Covers shamt 0-3 in one rule
#   This would expand to: allow shamt 0, 1, 2, 3

# Note: The 'allow' keyword is recognized by the parser but not fully
# implemented in codegen yet. It currently prints a warning.
# Full implementation will generate complementary patterns that forbid
# everything EXCEPT the allowed patterns.
