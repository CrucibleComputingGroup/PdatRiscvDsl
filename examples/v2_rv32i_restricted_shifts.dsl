version 2

# Example: Full RV32I but with restricted shift amounts
# Demonstrates sequential include/forbid operations

# Start with full RV32I instruction set
include RV32I

# Remove all shift instructions
forbid SLLI
forbid SRLI
forbid SRAI

# Add back only shifts with amount 0-7 (3-bit shifts)
include SLLI {shamt = 5'b00000}
include SLLI {shamt = 5'b00001}
include SLLI {shamt = 5'b00010}
include SLLI {shamt = 5'b00011}
include SLLI {shamt = 5'b00100}
include SLLI {shamt = 5'b00101}
include SLLI {shamt = 5'b00110}
include SLLI {shamt = 5'b00111}

include SRLI {shamt = 5'b00000}
include SRLI {shamt = 5'b00001}
include SRLI {shamt = 5'b00010}
include SRLI {shamt = 5'b00011}
include SRLI {shamt = 5'b00100}
include SRLI {shamt = 5'b00101}
include SRLI {shamt = 5'b00110}
include SRLI {shamt = 5'b00111}

include SRAI {shamt = 5'b00000}
include SRAI {shamt = 5'b00001}
include SRAI {shamt = 5'b00010}
include SRAI {shamt = 5'b00011}
include SRAI {shamt = 5'b00100}
include SRAI {shamt = 5'b00101}
include SRAI {shamt = 5'b00110}
include SRAI {shamt = 5'b00111}

# Result: Full RV32I with all shift instructions limited to 3-bit shift amounts (0-7)
# Upper 2 bits of shift amount are constrained to 00
