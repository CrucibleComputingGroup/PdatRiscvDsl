version 2

# Example: Allow ONLY SLLI with shift amounts 0-3
# This demonstrates the clean v2 syntax for your original use case

# Include only the specific SLLI patterns we want
include SLLI {shamt = 5'b00000}  # shift by 0
include SLLI {shamt = 5'b00001}  # shift by 1
include SLLI {shamt = 5'b00010}  # shift by 2
include SLLI {shamt = 5'b00011}  # shift by 3

# Result: Only these 4 SLLI variants are allowed
# All other instructions (including SLLI with shamt 4-31) are forbidden
