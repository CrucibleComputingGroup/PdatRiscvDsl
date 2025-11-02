version 2

# Example: Build a minimal ISA from scratch
# Start from empty set and add only what we need

# Add only the instructions we need
include ADD
include ADDI
include LW
include SW
include BEQ
include JAL

# Further restrict: ADDI only with small immediates
forbid ADDI
include ADDI {imm = 12'b000000000000}  # imm=0
include ADDI {imm = 12'b000000000001}  # imm=1
include ADDI {imm = 12'b000000000010}  # imm=2
include ADDI {imm = 12'b000000000011}  # imm=3
include ADDI {imm = 12'b000000000100}  # imm=4

# Result: Tiny ISA with 10 instruction variants
# - 1 ADD (all variants)
# - 5 ADDI (with imm 0-4)
# - 1 LW, 1 SW, 1 BEQ, 1 JAL
