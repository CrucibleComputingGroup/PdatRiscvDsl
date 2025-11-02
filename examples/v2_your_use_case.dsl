version 2

# Your original request: "restrict shift instructions to only shift by certain numbers"
# Using 5-bit quad-state logic: 5'b00xx restricts to shifts 0-3

# This is the exact use case you described!

# Allow ONLY SLLI with shift amounts 0-3 (5'b00xx pattern)
include SLLI {shamt = 5'b00000}  # 5'b00xx matches 0
include SLLI {shamt = 5'b00001}  # 5'b00xx matches 1
include SLLI {shamt = 5'b00010}  # 5'b00xx matches 2
include SLLI {shamt = 5'b00011}  # 5'b00xx matches 3

# This generates a positive assertion in SystemVerilog:
# assume (instruction is one of these 4 SLLI variants)
#
# All SLLI with shamt 4-31 are forbidden (not in the allowed set)
